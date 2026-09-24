"""Shared helpers for the System One samples: engine selection and an accuracy/latency/cost report.

Each sample runs with one of three engines, all answering the same decision schema:

- ``--engine jev``: TypeSafe's hosted Jev (needs ``TYPESAFE_API_KEY``).
- ``--engine kev``: a self-hosted Kev server (https://github.com/jaredpalmer/kev), which speaks the same
  ``/v1/systemone`` API. Start one with ``uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b
  --port 8009`` and point ``--kev-url`` (or ``KEV_BASE_URL``) at it; set ``KEV_API_KEY`` if the server needs one.
- ``--engine llm``: an LLM through ``LLMDecisionModel`` (needs AWS credentials for Amazon Bedrock).

Running more than one is the quickest way to compare them on your own data.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from strands.experimental.decisions import DecisionModel, DecisionResponse, DecisionState, LLMDecisionModel, Question
from strands.models import BedrockModel

T = TypeVar("T")

DEFAULT_LLM = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_KEV_URL = "http://127.0.0.1:8009"

# USD per million tokens (input, output), matching the companion baseline
# (team/designs/0020-system-one-decision-models-baseline.md): AWS Pricing API, us-west-2 on-demand, and
# docs.typesafe.ai/models (Jev bills input only). Retrieved 2026-09-24. Unknown models report no cost:
# a self-hosted Kev has no per-token price, so its cost is the instance time you pay for.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "jev": (0.042, 0.0),
    DEFAULT_LLM: (1.10, 5.50),
    "us.amazon.nova-micro-v1:0": (0.035, 0.14),
    # OpenAI's short-context Standard rate, which OpenAI states Bedrock matches in commercial regions.
    "global.openai.gpt-6-luna": (0.10, 0.50),
}


def parse_args(description: str) -> argparse.Namespace:
    """Parse the common ``--engine``, ``--llm-model``, ``--kev-url`` and ``--kev-model`` flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--engine", choices=["jev", "kev", "llm"], default="jev")
    parser.add_argument("--llm-model", default=DEFAULT_LLM, help="Bedrock model id for --engine llm")
    parser.add_argument(
        "--kev-url", default=os.environ.get("KEV_BASE_URL", DEFAULT_KEV_URL), help="Kev server for --engine kev"
    )
    parser.add_argument("--kev-model", default="kev-latest", help="model name sent to the Kev server")
    parser.add_argument(
        "--kev-temperature",
        type=float,
        default=1.0,
        help="temperature applied on the client (1.0 = as served); set a fitted value for a Kev server run with "
        "KEV_TEMPERATURE=1.0 (raw probabilities)",
    )
    return parser.parse_args()


def usd(model_id: str | None, usage: Mapping[str, int]) -> float:
    """Cost of one request's ``usage``; NaN when the model has no known price."""
    key = "jev" if (model_id or "").startswith("jev") else model_id
    price_in, price_out = PRICES_PER_MTOK.get(key or "", (math.nan, math.nan))
    return (usage.get("inputTokens", 0) * price_in + usage.get("outputTokens", 0) * price_out) / 1e6


class MeteredDecisionModel(DecisionModel):
    """Wrap a decision model and record the cost of every request, wherever an adapter makes it."""

    def __init__(self, inner: DecisionModel) -> None:
        self.inner = inner
        self.decisions = 0
        self.errors = 0
        self.cost = 0.0
        self.last: DecisionResponse | None = None  # the most recent response, for printing what an adapter saw

    @property
    def calibrated(self) -> bool:
        return self.inner.calibrated

    def get_config(self) -> Any:
        return self.inner.get_config()

    def update_config(self, **config: Any) -> None:
        self.inner.update_config(**config)

    async def _ask(self, state: DecisionState, questions: Mapping[str, Question], **kwargs: Any) -> DecisionResponse:
        try:
            response = await self.inner._ask(state, questions, **kwargs)
        except Exception:
            self.errors += 1
            raise
        self.decisions += 1
        self.cost += usd(response.model_id or self.inner.model_id, response.usage)
        self.last = response
        return response


def decision_model(args: argparse.Namespace) -> MeteredDecisionModel:
    """Return the decision engine selected on the command line, metered for the report."""
    if args.engine == "llm":
        return MeteredDecisionModel(LLMDecisionModel(BedrockModel(model_id=args.llm_model)))
    from strands.models import TypeSafeDecisionModel

    if args.engine == "kev":
        from strands.models.typesafe import KEV_MAX_STATE_PLUS_QUESTION_TOKENS

        return MeteredDecisionModel(
            TypeSafeDecisionModel(
                base_url=args.kev_url,
                api_key=os.environ.get("KEV_API_KEY") or None,
                model_id=args.kev_model,
                temperature=args.kev_temperature,
                max_state_plus_question_tokens=KEV_MAX_STATE_PLUS_QUESTION_TOKENS,
                max_request_tokens=None,
                client_args={"timeout": 120},  # generous for CPU-served checkpoints
            )
        )
    return MeteredDecisionModel(TypeSafeDecisionModel())


class Report:
    """Collect per-case latency and correctness; print accuracy, p50/p95 latency, and cost per 1k decisions.

    A case whose decision request failed is counted as an error, not scored: an adapter that declines on an error
    (a router default, a guard's fail-closed Confirm) can otherwise look correct without any decision being made.
    """

    def __init__(self, engine: str, metered: MeteredDecisionModel) -> None:
        self.engine = engine
        self.metered = metered
        self.latencies: list[float] = []
        self.correct: list[bool] = []
        self.errored: list[str] = []
        calibrated = (
            "calibrated: confidence gates apply"
            if metered.calibrated
            else ("NOT calibrated: probabilities are one-hot and confidence is None, so confidence gates are off")
        )
        print(f"engine {engine}: {calibrated}")

    async def time(self, call: Callable[[], Awaitable[T]]) -> T:
        started = time.perf_counter()
        value = await call()
        self.latencies.append(time.perf_counter() - started)
        return value

    def check(self, label: str, expected: Any, actual: Any, detail: str = "", *, errors_before: int = -1) -> None:
        """Record one case; ``expected`` may be a set of acceptable answers.

        Pass ``errors_before=report.metered.errors`` captured before the case ran: if the engine errored during
        the case, it is recorded as an error rather than scored.
        """
        if 0 <= errors_before < self.metered.errors:
            self.errored.append(label)
            print(f"  [ERROR] {label}: the decision request failed; not scored {detail}")
            return
        ok = actual in expected if isinstance(expected, (set, frozenset)) else expected == actual
        self.correct.append(ok)
        print(f"  [{'ok' if ok else 'MISS'}] {label}: expected={expected!r} got={actual!r} {detail}")

    def summary(self) -> None:
        """Print the result line and exit non-zero if any decision request failed."""
        scored = len(self.correct)
        accuracy = sum(self.correct) / max(1, scored)
        print(
            f"\n{self.engine}: {sum(self.correct)}/{scored} correct ({accuracy:.0%}), "
            f"p50 {_percentile(self.latencies, 50):.2f}s / {_p95(self.latencies)}, "
            f"{self._cost()} ({self.metered.decisions} decisions, {self.metered.errors} errors)"
        )
        print(
            f"  {scored} labelled cases: this shows behaviour, not accuracy. For accuracy with confidence intervals,"
            " see team/designs/0020-system-one-decision-models-baseline.md."
        )
        if self.metered.errors or self.errored:
            print(f"  {self.metered.errors} decision request(s) failed; see the errors above.", file=sys.stderr)
            raise SystemExit(1)

    def _cost(self) -> str:
        if not self.metered.decisions:
            return "decision cost not metered (no DecisionModel requests)"
        per_1k = 1000 * self.metered.cost / self.metered.decisions
        return "no per-token price (self-hosted)" if math.isnan(per_1k) else f"${per_1k:.3f} per 1k decisions"


def distribution(probabilities: Mapping[Any, float], top: int = 3) -> str:
    """Format the most probable answers, e.g. ``billing 0.65, account 0.24, technical 0.06``."""
    ranked = sorted(probabilities.items(), key=lambda item: -item[1])[:top]
    return ", ".join(f"{key} {value:.2f}" for key, value in ranked)


def _p95(values: list[float]) -> str:
    if len(values) < 20:
        return f"max {max(values):.2f}s (p95 needs 20+ samples)" if values else "max n/a"
    return f"p95 {_percentile(values, 95):.2f}s"


def _percentile(values: list[float], pct: int) -> float:
    if len(values) < 2:
        return values[0] if values else math.nan
    return statistics.quantiles(values, n=100, method="inclusive")[pct - 1]
