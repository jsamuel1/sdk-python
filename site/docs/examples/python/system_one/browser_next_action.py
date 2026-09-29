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

`--fast-path` runs the same idea inside an agent loop instead: a Bedrock agent shops on a simulated
site with `search` and `click` tools, and `FastPath` serves the steps that are one fixed control
(close a popup, open the cart, check out, place the order) from the decision model, so the LLM only
runs for steps that need it (a search query, a product choice under a price limit, finishing). It
runs every task twice, all-LLM and with FastPath, and reports the share of steps System One served,
task success checked in code, and cost.

Usage:
    python browser_next_action.py                # Jev (TYPESAFE_API_KEY) + Bedrock for escalations
    python browser_next_action.py --engine kev   # self-hosted Kev (--kev-url, default KEV_BASE_URL)
    python browser_next_action.py --engine llm   # same questions on an LLM (every step escalates)
    python browser_next_action.py --fast-path    # agent loop: all-LLM arm vs FastPath arm (calibrated engine)

Moving to a real browser: activate the agent's tab (headless browsers throttle background tabs, so
fade-in menus never become observable), keep fixture dates in the future, and expect accessible names
to differ between browsers. See the Gemma replication notes linked from the design's baseline.
"""

from __future__ import annotations

import asyncio
import re
from typing import Literal

from _common import Report, decision_model, distribution, parse_args, usd
from pydantic import create_model
from strands import Agent, tool
from strands.experimental.decisions import Choice, FastPath, ToolCall, YesNo
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


# ---- --fast-path: FastPath inside an agent loop -------------------------------------------------

CATALOG = {
    "usb-c cable": {
        "p1": ("Apple USB-C Cable 1m", 19.00),
        "p2": ("Anker USB-C Cable 6ft", 12.99),
        "p3": ("Belkin USB-C Cable 2m", 16.50),
    },
    "hdmi cable": {
        "p4": ("AmazonBasics HDMI Cable 6ft", 7.49),
        "p5": ("Belkin HDMI 2.1 Cable 2m", 24.99),
        "p6": ("Cable Matters HDMI Cable 3ft", 8.99),
    },
    "aa batteries": {
        "p7": ("Duracell AA 8-pack", 9.99),
        "p8": ("Energizer AA 4-pack", 5.49),
    },
}
TASKS = [
    ("Buy one USB-C cable that costs under $15, then place the order.", "p2"),
    ("Buy the cheapest HDMI cable you can find, then place the order.", "p4"),
    ("Buy an 8-pack of AA batteries, then place the order.", "p7"),
]
# The closed action set: controls whose click never needs a judgment beyond "is this the next step".
FAST_ACTIONS = {
    "close_popup": ToolCall("click", {"element": "close-popup"}, description="Close the 'Added to cart' popup"),
    "open_cart": ToolCall("click", {"element": "cart"}, description="Open the cart after the item was added"),
    "checkout": ToolCall("click", {"element": "checkout"}, description="Proceed to checkout from the cart"),
    "place_order": ToolCall("click", {"element": "place-order"}, description="Place the order on the checkout page"),
}
FAST_INSTRUCTIONS = (
    "An agent drives a web shop toward `request`; `latest_tool_result` is the page it is on. Which single control "
    "should it click next? Choose `other` if the page has no such control, if the step needs a search query or a "
    "product choice, or if the order is already placed."
)
SHOPPER_PROMPT = (
    "You shop on a web store with the search and click tools. Each tool returns the page you are on and its "
    "clickable element ids. Add exactly the item the user asked for, then open the cart, check out, and place the "
    "order. Reply 'done' once the order is placed."
)
MAX_TURNS = 20


class Shop:
    """A deterministic web shop. Every tool result is the new page and its clickable element ids."""

    def __init__(self) -> None:
        self.page, self.results, self.cart, self.orders = "home", {}, [], []
        self.popup = False

    def render(self) -> str:
        if self.popup:
            return "page: popup 'Added to cart'. elements: close-popup"
        if self.page == "results":
            items = "; ".join(f"add-{pid}: add '{name} - ${price:.2f}'" for pid, (name, price) in self.results.items())
            return f"page: search results. elements: {items}; cart"
        if self.page == "cart":
            lines = ", ".join(CATALOG_BY_ID[pid][0] for pid in self.cart) or "empty"
            return f"page: cart ({lines}). elements: checkout" if self.cart else "page: cart (empty)"
        if self.page == "checkout":
            return "page: checkout, review your order. elements: place-order"
        if self.page == "confirmation":
            return "page: order placed. Thank you! elements: none"
        return "page: home. elements: none; use search"

    def search(self, query: str) -> str:
        self.popup = False
        self.page = "results"
        self.results = next((items for key, items in CATALOG.items() if key in query.lower()), {})
        return self.render()

    def click(self, element: str) -> str:
        handler = {
            "close-popup": self._close_popup,
            "cart": lambda: self._goto("cart", self.page == "results" and not self.popup),
            "checkout": lambda: self._goto("checkout", self.page == "cart" and bool(self.cart)),
            "place-order": self._place_order,
        }.get(element)
        if handler is None and element.startswith("add-") and element[4:] in self.results and not self.popup:
            self.cart.append(element[4:])
            self.popup = True
            return self.render()
        if handler is None or not handler():
            return f"error: no clickable element '{element}' here. {self.render()}"
        return self.render()

    def _close_popup(self) -> bool:
        if not self.popup:
            return False
        self.popup = False
        return True

    def _goto(self, page: str, allowed: bool) -> bool:
        if allowed:
            self.page = page
        return allowed

    def _place_order(self) -> bool:
        if self.page != "checkout":
            return False
        self.orders.append(list(self.cart))
        self.cart, self.page = [], "confirmation"
        return True


CATALOG_BY_ID = {pid: item for items in CATALOG.values() for pid, item in items.items()}


def shop_tools(shop: Shop) -> list:
    @tool
    def search(query: str) -> str:
        """Search the store; returns the results page."""
        return shop.search(query)

    @tool
    def click(element: str) -> str:
        """Click an element id on the current page; returns the new page."""
        return shop.click(element)

    return [search, click]


def served_by_decision(message: dict) -> bool:
    return message.get("metadata", {}).get("custom", {}).get("strands", {}).get("source") == "decision"


async def run_task(llm_model: str, goal: str, fast_path: FastPath | None) -> dict:
    shop = Shop()
    agent = Agent(
        model=BedrockModel(model_id=llm_model),
        system_prompt=SHOPPER_PROMPT,
        tools=shop_tools(shop),
        plugins=[fast_path] if fast_path else [],
        callback_handler=None,
    )
    result = await agent.invoke_async(goal, limits={"turns": MAX_TURNS})
    steps = [m for m in agent.messages if m["role"] == "assistant" and any("toolUse" in b for b in m["content"])]
    usage = result.metrics.accumulated_usage
    return {
        "steps": len(steps),
        "served": sum(served_by_decision(m) for m in steps),
        "orders": shop.orders,
        "stop": result.stop_reason,
        "llm_usd": usd(llm_model, usage),
        "llm_tokens": usage["totalTokens"],
    }


async def fast_path_arms(args, engine) -> None:
    if not engine.calibrated:
        raise SystemExit(f"--fast-path needs a calibrated engine; {args.engine} is not (FastPath refuses it).")
    print(f"FastPath agent loop ({args.engine} + {args.llm_model}), {len(TASKS)} tasks")
    totals = {
        arm: {"steps": 0, "served": 0, "ok": 0, "usd": 0.0, "decision_usd": 0.0, "seconds": 0.0}
        for arm in ("llm", "fast_path")
    }
    for goal, expected in TASKS:
        for arm in totals:
            fast_path = (
                FastPath(engine, FAST_ACTIONS, min_confidence=args.min_confidence, instructions=FAST_INSTRUCTIONS)
                if arm == "fast_path"
                else None
            )
            cost_before, errors_before = engine.cost, engine.errors
            started = asyncio.get_running_loop().time()
            run = await run_task(args.llm_model, goal, fast_path)
            seconds = asyncio.get_running_loop().time() - started
            ok = run["orders"] == [[expected]]
            decision_usd = engine.cost - cost_before
            row = totals[arm]
            row["steps"] += run["steps"]
            row["served"] += run["served"]
            row["ok"] += ok
            row["usd"] += run["llm_usd"] + decision_usd
            row["decision_usd"] += decision_usd
            row["seconds"] += seconds
            print(
                f"  [{'ok' if ok else 'MISS'}] {arm:9} {goal[:40]!r}: steps={run['steps']} served={run['served']} "
                f"orders={run['orders']} stop={run['stop']} llm_tokens={run['llm_tokens']} "
                f"cost=${run['llm_usd'] + decision_usd:.5f} decision_errors={engine.errors - errors_before} "
                f"{seconds:.1f}s"
            )
    print()
    for arm, row in totals.items():
        share = row["served"] / max(1, row["steps"])
        print(
            f"{arm:9}: success {row['ok']}/{len(TASKS)}, steps {row['steps']}, served by System One {row['served']} "
            f"({share:.0%}), cost ${row['usd']:.4f} (decisions ${row['decision_usd']:.5f}), wall {row['seconds']:.1f}s"
        )
    print(f"  floor {args.min_confidence}; decision requests {engine.decisions}, errors {engine.errors}")
    if engine.errors:
        raise SystemExit(1)


def _fast_path_flags(parser) -> None:
    parser.add_argument("--fast-path", action="store_true", help="run the agent-loop all-LLM vs FastPath arms")
    parser.add_argument("--min-confidence", type=float, default=CONFIDENCE_FLOOR, help="FastPath floor")


async def main() -> None:
    args = parse_args(__doc__.splitlines()[1], _fast_path_flags)
    engine = decision_model(args)
    if args.fast_path:
        await fast_path_arms(args, engine)
        return
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
