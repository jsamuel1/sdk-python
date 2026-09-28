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

Placement (c), ``--swarm``: a ``Swarm`` whose triage agent is an LLM (``--llm-model`` on Amazon Bedrock, so
this arm also needs AWS credentials). It runs twice on the same tickets. In the LLM arm, the triage agent
hands off by calling ``handoff_to_agent``, as swarms do today. In the decision arm, it only summarises the
ticket and ``DecisionHandoffStrategy`` picks the team. The strategy also runs after each specialist, which
should answer ``complete``, so the arm makes two decisions per ticket. Both arms report handoff accuracy,
extra hops past the specialist, latency, and cost per 1,000 tickets (the triage agent's LLM tokens plus the
decisions). The specialists are code stand-ins.

Usage:
    python support_triage.py                # Jev (TYPESAFE_API_KEY)
    python support_triage.py --engine kev   # self-hosted Kev (--kev-url, default KEV_BASE_URL)
    python support_triage.py --engine llm   # same schema on an LLM (Amazon Bedrock)
    python support_triage.py --swarm        # swarm handoff: LLM handoff_to_agent vs Jev (also needs AWS)
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections.abc import AsyncGenerator
from typing import Annotated, Any, Literal

from _common import Report, decision_model, distribution, parse_args, usd
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.experimental.decisions import (
    COMPLETE_OPTION,
    DECISION_STATE_KEY,
    Choice,
    DecisionAgent,
    DecisionHandoffStrategy,
    DecisionSchema,
    YesNo,
    when_below,
    when_choice,
    when_yes,
)
from strands.models import BedrockModel, Model
from strands.multiagent import GraphBuilder, Swarm

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


TEAMS = {
    "billing": "Charges, invoices, refunds, payment methods",
    "technical": "Bugs, outages, errors, integrations",
    "account": "Login, passwords, MFA, profile settings",
    "general": "Messages that raise more than one independent problem, or fit no other team",
}
TRIAGE_SUMMARISE = "You triage customer support messages. Restate the customer's problem in one sentence."
TRIAGE_HANDOFF = (
    f"{TRIAGE_SUMMARISE} Then call handoff_to_agent to pass the message to the team that should handle it: "
    "billing, technical, or account, or general when the message raises more than one independent problem."
)
TRIAGE_NO_HANDOFF = f"{TRIAGE_SUMMARISE} Do not hand off; another system routes the message."


class _TeamReply(Model):
    """A Model that replies with the team's name: a code stand-in for a specialist agent (no LLM call)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_config(self) -> Any:
        return {"model_id": f"stand-in-{self.name}"}

    def update_config(self, **config: Any) -> None:
        pass

    def structured_output(self, output_model: Any, prompt: Any, system_prompt: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def stream(
        self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": self.name}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def build_swarm(llm_model: str, strategy: DecisionHandoffStrategy | None) -> Swarm:
    triage = Agent(
        name="triage",
        model=BedrockModel(model_id=llm_model),
        system_prompt=TRIAGE_HANDOFF if strategy is None else TRIAGE_NO_HANDOFF,
        callback_handler=None,
    )
    teams = [
        Agent(name=name, description=description, model=_TeamReply(name), callback_handler=None)
        for name, description in TEAMS.items()
    ]
    return Swarm([triage, *teams], handoff_strategy=strategy, max_handoffs=3, max_iterations=3)


async def swarm_arm(label: str, swarm: Swarm, engine, llm_model: str) -> tuple[int, int]:
    """Run every case through ``swarm``; return (decision requests that failed, cases the LLM routed itself)."""
    correct, declined, llm_routed, extra_hops, latencies, cost = 0, 0, 0, 0, [], 0.0
    errors_before, decisions_before, decision_cost_before = engine.errors, engine.decisions, engine.cost
    print(f"\nswarm, {label}")
    for message, expected in CASES:
        started = time.perf_counter()
        result = await swarm.invoke_async(message)
        latencies.append(time.perf_counter() - started)
        routed = result.node_history[1].node_id if len(result.node_history) > 1 else "triage"
        triage = result.results["triage"]
        cost += usd(llm_model, triage.accumulated_usage)
        # A specialist that finishes is also asked where to go next; any hop past it is an extra handoff.
        extra_hops += max(0, len(result.node_history) - 2)
        decision = getattr(triage.result, "state", {}).get(DECISION_STATE_KEY)
        detail = ""
        if decision is not None:
            answer = decision.answers["next"]
            detail = f"confidence={answer.confidence} [{distribution(answer.probabilities)}]"
            declined += routed == "triage" and decision.output.next != COMPLETE_OPTION
        elif swarm.handoff_strategy is not None and routed != "triage":
            llm_routed += 1
            detail = "routed by the LLM's own handoff_to_agent call"
        ok = routed == expected
        correct += ok
        print(f"  [{'ok' if ok else 'MISS'}] {message[:48]}: expected={expected!r} got={routed!r} {detail}")
    cost += engine.cost - decision_cost_before
    per_1k = "n/a (unpriced model)" if math.isnan(cost) else f"${1000 * cost / len(CASES):.3f} per 1k tickets"
    print(
        f"  {label}: {correct}/{len(CASES)} correct, {declined} declined, {extra_hops} extra hops, "
        f"p50 {statistics.median(latencies):.2f}s / max {max(latencies):.2f}s per ticket, {per_1k} "
        f"({engine.decisions - decisions_before} decisions)"
    )
    return engine.errors - errors_before, llm_routed


async def swarm_arms(engine, args) -> None:
    await swarm_arm("LLM handoff_to_agent", build_swarm(args.llm_model, None), engine, args.llm_model)
    strategy = DecisionHandoffStrategy(engine, min_confidence=MIN_CONFIDENCE if engine.calibrated else None)
    errors, llm_routed = await swarm_arm(
        f"DecisionHandoffStrategy ({args.engine})", build_swarm(args.llm_model, strategy), engine, args.llm_model
    )
    if llm_routed:
        print(f"  {llm_routed} ticket(s) were routed by the LLM despite its instructions; its own handoff wins.")
    if errors:
        print(f"  {errors} decision request(s) failed; see the errors above.")
        raise SystemExit(1)


def _swarm_flag(parser) -> None:
    parser.add_argument("--swarm", action="store_true", help="compare swarm handoff arms (needs AWS credentials)")


async def main() -> None:
    args = parse_args(__doc__.splitlines()[1], _swarm_flag)
    engine = decision_model(args)
    if args.swarm:
        await swarm_arms(engine, args)
        return
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
