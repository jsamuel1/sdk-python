#!/usr/bin/env python3
"""
# Browser use: pick the next element, check the goal, escalate when unsure

Use case: driving computer and browser use. Code extracts the actionable elements from the page
(here a fixed accessibility snapshot, so the sample needs no browser). One request asks a Choice
over the element ids plus two YesNo questions: does the page meet the goal, and does this step need
multi-step reasoning? Confident, simple steps act directly. Anything else goes to the reasoning
computer-use agent, the "System Two" escalation path.

"Select, don't generate": the model never writes a selector, it picks one of the ids code found, and
the escalation path is constrained to the same ids.

"Keep facts in code": the goal is a list of requirements that code checks against the page. The
model's goal_reached answer is a hint, and the step is only DONE when code confirms every requirement.

Usage:
    python browser_next_action.py                # Jev (TYPESAFE_API_KEY) + Bedrock for escalations
    python browser_next_action.py --engine kev   # self-hosted Kev (--kev-url, default KEV_BASE_URL)
    python browser_next_action.py --engine llm   # same questions on an LLM (every step escalates)

Moving to a real browser: activate the agent's tab (headless browsers throttle background tabs, so
fade-in menus never become observable), keep fixture dates in the future, and expect accessible names
to differ between browsers. See the Gemma replication notes linked from the design's baseline.
"""

from __future__ import annotations

import asyncio
import re
from typing import Literal

from _common import Report, decision_model, distribution, parse_args
from pydantic import create_model
from strands import Agent
from strands.experimental.decisions import Choice, YesNo
from strands.models import BedrockModel

# Illustrative floors, tuned on these few pages only; tune per engine on your own traffic. Kev's probabilities
# are flatter than Jev's (a fitted temperature of 2.41 on Kev-4B), so the same floor escalates more often on Kev.
CONFIDENCE_FLOOR = 0.7
NEEDS_REASONING_ABOVE = 0.5

CART_GOAL = {"text": "Add a USB-C cable under $15 to the cart", "in_cart": r"USB-C Cable", "max_price": 15.0}
PAGES = [
    {
        "goal": CART_GOAL,
        "page": "search results for 'usb-c cable'",
        "elements": {
            "e1": "link 'Anker USB-C Cable 6ft - $12.99'",
            "e2": "button 'Add to cart' beside 'Anker USB-C Cable 6ft - $12.99'",
            "e3": "button 'Add to cart' beside 'Apple USB-C Cable - $19.00'",
            "e4": "input 'Search'",
        },
        "expected": ("e2", False),
    },
    {
        "goal": CART_GOAL,
        "page": "cart: 'Anker USB-C Cable 6ft - $12.99' x1, subtotal $12.99",
        "elements": {"e1": "button 'Checkout'", "e2": "button 'Remove'", "e3": "link 'Continue shopping'"},
        "expected": ("none", True),
    },
    {
        # Negative case: a cable is in the cart, but it breaks the price requirement, so the goal is NOT met.
        "goal": CART_GOAL,
        "page": "cart: 'Apple USB-C Cable - $19.00' x1, subtotal $19.00",
        "elements": {"e1": "button 'Checkout'", "e2": "button 'Remove'", "e3": "link 'Continue shopping'"},
        "expected": ({"e2", "e3"}, False),
    },
    {
        "goal": {"text": "Change the account email to ops@example.com"},
        "page": "settings: 'Email: old@example.com'",
        "elements": {
            "e1": "button 'Edit' beside 'Email'",
            "e2": "button 'Edit' beside 'Phone'",
            "e3": "link 'Delete account'",
        },
        "expected": ("e1", False),
    },
    {
        # Two near-identical options: whichever the engine prefers, this page is where it may be unsure enough to
        # escalate. Either room is acceptable.
        "goal": {"text": "Book the 9am meeting room"},
        "page": "rooms",
        "elements": {"e1": "button 'Book' in row 'Room A 9:00'", "e2": "button 'Book' in row 'Room B 9:00'"},
        "expected": ({"e1", "e2"}, False),
    },
]


def goal_met_in_code(goal: dict, page: str) -> bool | None:
    """Check the goal's requirements against the page text. None when the goal has no code-checkable requirement."""
    if "in_cart" not in goal:
        return None
    if not page.startswith("cart:") or not re.search(goal["in_cart"], page):
        return False
    prices = [float(p) for p in re.findall(r"\$(\d+(?:\.\d+)?)", page)]
    return bool(prices) and max(prices) <= goal["max_price"]


def questions(elements: dict[str, str]) -> dict:
    return {
        "action": Choice(
            "Which element in `elements` should be clicked next to make progress toward `goal` on `page`?",
            options={**{key: None for key in elements}, "none": "No element is needed or none helps"},
        ),
        "goal_reached": YesNo("Does `page` show that every requirement of `goal` is met?"),
        "needs_reasoning": YesNo(
            "Does choosing the next step need multi-step planning, arithmetic, or comparing several constraints?"
        ),
    }


async def escalate(model_id: str, state: dict, elements: dict[str, str]) -> str:
    """The System Two path, constrained to the page's element ids: it cannot answer with an id that is not there."""
    step = create_model("NextStep", element=(Literal[(*elements, "none")], ...))
    # A fresh agent per step: each step is independent, and a shared agent would carry earlier pages as history.
    agent = Agent(
        model=BedrockModel(model_id=model_id),
        system_prompt="You drive a web browser. Pick the element id to click next, or 'none' if no click helps.",
        callback_handler=None,
    )
    result = await agent.invoke_async(str(state), structured_output_model=step)
    return result.structured_output.element


def fast_path_allowed(engine, action, needs_reasoning, unverified_goal_claim: bool) -> bool:
    # An uncalibrated engine has no confidence: None means unsure, never sure, so every step escalates.
    if not engine.calibrated or action.confidence is None:
        return False
    # The model says the goal is reached but code has no requirement to check it against: System Two decides.
    if unverified_goal_claim:
        return False
    return action.confidence >= CONFIDENCE_FLOOR and needs_reasoning.probability < NEEDS_REASONING_ABOVE


async def main() -> None:
    args = parse_args(__doc__.splitlines()[1])
    engine = decision_model(args)
    report = Report(args.engine, engine)
    escalations = 0
    print(f"browser next action ({args.engine})")
    for page in PAGES:
        state = {"goal": page["goal"]["text"], "page": page["page"], "elements": page["elements"]}
        before = report.metered.errors
        response = await report.time(lambda state=state: engine.ask(state, questions(state["elements"])))
        action, reached, reasoning = response["action"], response["goal_reached"], response["needs_reasoning"]
        in_code = goal_met_in_code(page["goal"], page["page"])
        # Only code marks a step DONE. The model's goal_reached is a hint: a claim code cannot check escalates.
        done = in_code is True
        unverified_goal_claim = in_code is None and reached.probability >= 0.5
        path = "done"
        if done:
            decision = "none"
        elif fast_path_allowed(engine, action, reasoning, unverified_goal_claim):
            path, decision = "fast", action.choice
        else:
            path, escalations = "escalated", escalations + 1
            decision = await escalate(args.llm_model, state, page["elements"])
        expected_action, expected_done = page["expected"]
        detail = (
            f"path={path} confidence={action.confidence} [{distribution(action.probabilities)}]"
            f" needs_reasoning p={reasoning.probability:.2f}"
        )
        report.check(page["page"][:40], expected_action, decision, detail, errors_before=before)
        report.check(
            "  goal met",
            expected_done,
            done,
            f"code={in_code} model p={reached.probability:.2f}",
            errors_before=before,
        )
    print(f"\nescalated to the computer-use agent: {escalations}/{len(PAGES)} steps")
    report.summary()


if __name__ == "__main__":
    asyncio.run(main())
