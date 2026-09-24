#!/usr/bin/env python3
"""
# Support triage: a System One model as the front door and as a graph router

Use case: dynamic multi-agent routing and handoffs.

Placement (a), front door: a ``DecisionAgent`` sees every incoming message first, dispatches
confident single-issue requests straight to a specialist, and hands anything uncertain, unmatched,
or multi-issue to a general reasoning agent. Placement (b), subagent: the same schema drives a
``Graph`` through ``when_choice`` / ``when_yes`` / ``when_below`` edges, with the same outcomes.

Both placements use code stand-ins for the specialists and the fallback, so ``--engine jev`` needs only
``TYPESAFE_API_KEY``. In an app these are full agents with tools.

Usage:
    python support_triage.py                # Jev (TYPESAFE_API_KEY)
    python support_triage.py --engine kev   # self-hosted Kev (--kev-url, default KEV_BASE_URL)
    python support_triage.py --engine llm   # same schema on an LLM (Amazon Bedrock)
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal

from _common import Report, decision_model, distribution, parse_args
from strands.agent.agent_result import AgentResult
from strands.experimental.decisions import (
    DECISION_STATE_KEY,
    Choice,
    DecisionAgent,
    DecisionSchema,
    YesNo,
    when_below,
    when_choice,
    when_yes,
)
from strands.multiagent import GraphBuilder

SPECIALISTS = ("billing", "technical", "account")
# Illustrative, not tuned: tune per engine on your own traffic.
MIN_CONFIDENCE = 0.6


class Triage(DecisionSchema):
    department: Annotated[
        Literal["billing", "technical", "account"] | None,
        Choice(
            "Which team should handle the customer's message?",
            options={
                "billing": "Charges, invoices, refunds, payment methods",
                "technical": "Bugs, outages, errors, integrations",
                "account": "Login, passwords, MFA, profile settings",
            },
        ),
    ]
    multi_issue: Annotated[bool, YesNo("Does the message raise more than one independent problem?")]


CASES = [
    ("You billed my card twice for the March invoice.", "billing"),
    ("The dashboard throws a 500 error whenever I export a report.", "technical"),
    ("I reset my password but the MFA code never arrives.", "account"),
    # Two independent issues: no single specialist can resolve them, so the request goes to the fallback.
    ("I was double-charged AND I can't log in to fix it.", "general"),
    # From Kev's README: a late delivery, a wrong size and a double charge. It leans billing, but it is multi-issue.
    ("Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "general"),
]


def canned(name: str):
    """A code route: no LLM call at all for the confident, well-understood path."""
    return lambda decision, prompt: "general" if decision.output.multi_issue else name


def _describe(decision) -> str:
    department = decision.answers["department"]
    multi = decision.answers["multi_issue"]
    return (
        f"confidence={department.confidence} [{distribution(department.probabilities)}] "
        f"multi_issue p={multi.probability:.2f}"
    )


async def front_door(engine, report: Report) -> None:
    support = DecisionAgent(
        engine,
        Triage,
        route_on="department",
        routes={name: canned(name) for name in SPECIALISTS},
        # Only a calibrated engine may gate on confidence; the LLM engine routes on its answer alone.
        min_confidence=MIN_CONFIDENCE if engine.calibrated else None,
        fallback=lambda decision, prompt: "general",
    )
    print(f"front door ({report.engine})")
    for message, expected in CASES:
        before = report.metered.errors
        result = await report.time(lambda message=message: support.invoke_async(message))
        report.check(
            message[:48],
            expected,
            str(result).strip(),
            _describe(result.state[DECISION_STATE_KEY]),
            errors_before=before,
        )


class _Stand_in:
    """A graph node that replies with its name: a code stand-in for a specialist agent (no LLM call)."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def invoke_async(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
        return AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": self.name}]},
            metrics=None,
            state={},
        )

    def __call__(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
        return asyncio.run(self.invoke_async(prompt))

    async def stream_async(self, prompt: Any = None, **kwargs: Any):
        yield {"result": await self.invoke_async(prompt)}


MULTI_ISSUE = when_yes("router", "multi_issue")
UNSURE = when_below("router", "department", MIN_CONFIDENCE)
NO_MATCH = when_choice("router", "department", "none")


def _single_issue_choice(name: str):
    """Specialist edge: confidently this department AND not multi-issue."""
    chose = when_choice("router", "department", name, min_confidence=MIN_CONFIDENCE)

    def condition(state, *, invocation_state):
        return chose(state, invocation_state=invocation_state) and not MULTI_ISSUE(
            state, invocation_state=invocation_state
        )

    return condition


def _needs_general(state, *, invocation_state) -> bool:
    """Fallback edge: unsure, confidently none of the specialists, or multi-issue. Exactly one edge fires."""
    return any(check(state, invocation_state=invocation_state) for check in (UNSURE, NO_MATCH, MULTI_ISSUE))


def build_graph(engine):
    builder = GraphBuilder()
    builder.add_node(DecisionAgent(engine, Triage, name="router"), "router")
    for name in (*SPECIALISTS, "general"):
        builder.add_node(_Stand_in(name), name)
    for name in SPECIALISTS:
        builder.add_edge("router", name, condition=_single_issue_choice(name))
    builder.add_edge("router", "general", condition=_needs_general)
    builder.set_entry_point("router")
    return builder.build()


async def graph_router(engine, report: Report) -> None:
    graph = build_graph(engine)
    print(f"\ngraph router ({report.engine})")
    for message, expected in CASES:
        before = report.metered.errors
        result = await report.time(lambda message=message: graph.invoke_async(message))
        ran = sorted(set(result.results) - {"router"})
        decision = result.results["router"].result.state[DECISION_STATE_KEY]  # the canonical decision location
        report.check(message[:48], [expected], ran, _describe(decision), errors_before=before)


async def main() -> None:
    args = parse_args(__doc__.splitlines()[1])
    engine = decision_model(args)
    report = Report(args.engine, engine)
    if not engine.calibrated:
        print(
            "  The LLM engine gives no confidence, so routing rests on its multi_issue answer alone. That answer is a"
            " one-hot bool (0 or 1), and it can flip between runs on borderline tickets: LLM stated confidence"
            " saturates, which is why the confidence gate is off here."
        )
        await front_door(engine, report)
    else:
        await front_door(engine, report)
        await graph_router(engine, report)
    report.summary()


if __name__ == "__main__":
    asyncio.run(main())
