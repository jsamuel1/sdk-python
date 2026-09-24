#!/usr/bin/env python3
"""
# Mixture of models: a System One model picks the model tier per request

Use case: model selection. ``DecisionStrategy`` plugs into ``ModelRouter``: one Choice over the
candidates' descriptions picks the least capable model that can handle the request. Below
``min_confidence`` it declines, and the router serves its first (safe default) candidate. That
decline on ambiguity is what an LLM classifier cannot do: its answer carries no confidence.

Placement: embedded strategy, the agent never sees the decision.

Use ``DecisionStrategy`` with a calibrated model (Jev, Kev). With only an LLM, use the SDK's
``ClassifierStrategy``, which is what ``--engine llm`` runs here as the status-quo comparison.

Usage:
    python mixture_of_models.py                # Jev (TYPESAFE_API_KEY) + Amazon Bedrock candidates
    python mixture_of_models.py --engine kev   # self-hosted Kev classifies instead of Jev
    python mixture_of_models.py --engine llm   # ClassifierStrategy on Haiku (the status quo)
"""

from __future__ import annotations

import asyncio

from _common import PRICES_PER_MTOK, Report, decision_model, distribution, parse_args
from strands import Agent
from strands.experimental.decisions import DecisionStrategy
from strands.models import BedrockModel, ClassifierStrategy, ModelRouter, RoutingCandidate, RoutingContext

ROUTINE = "us.amazon.nova-micro-v1:0"
COMPLEX = "us.anthropic.claude-sonnet-4-6"
# The expected outcome for a request the strategy should not decide: decline, so the router serves its default.
DECLINE = "declined"
# The floor is illustrative, not tuned: tune it per engine on your own traffic (Kev's probabilities are flatter
# than Jev's, so the same number means something different).
MIN_CONFIDENCE = 0.7
# Rough generation cost of one answer, for the routing economics line: ~600 input and ~400 output tokens.
GEN_TOKENS = (600, 400)
# Sonnet 4.6: AWS Pricing API, us-west-2 on-demand (US profile), retrieved 2026-09-27.
PRICES = {**PRICES_PER_MTOK, COMPLEX: (3.30, 16.50)}

CASES = [
    ("What is the capital of Australia?", ROUTINE),
    ("Summarise this sentence in five words: the meeting moved to Thursday at noon.", ROUTINE),
    ("Prove that there are infinitely many primes, then write the proof in Lean 4.", COMPLEX),
    ("Design a sharded rate limiter for 50k RPS with exactly-once accounting; justify each tradeoff.", COMPLEX),
    # Ambiguous on purpose: a quick opinion or a design question, depending on context the prompt lacks. The right
    # answer is to decline and let the router serve its capable default; picking a tier is a miss. Jev splits roughly
    # 50/50 here (confidence near 0.1). An LLM classifier has no confidence, so it cannot decline and always misses.
    ("What's the best database for my startup?", DECLINE),
]


def router(strategy) -> ModelRouter:
    return ModelRouter(
        models=[
            # First candidate is the default when the strategy declines, so make it the capable one.
            RoutingCandidate(
                BedrockModel(model_id=COMPLEX, max_tokens=512),
                name="complex",
                description="Multi-step reasoning, proofs, code generation, system design",
            ),
            RoutingCandidate(
                BedrockModel(model_id=ROUTINE, max_tokens=512),
                name="routine",
                description="Direct factual questions, short summaries, extraction",
            ),
        ],
        strategy=strategy,
    )


def strategy_for(args, engine):
    if engine.calibrated:
        return DecisionStrategy(engine, min_confidence=MIN_CONFIDENCE)
    # An uncalibrated model cannot gate on confidence, so compare against what the SDK ships for LLMs.
    return ClassifierStrategy(model=BedrockModel(model_id=args.llm_model))


def gen_cost(model_id: str) -> float:
    price_in, price_out = PRICES[model_id]
    return (GEN_TOKENS[0] * price_in + GEN_TOKENS[1] * price_out) / 1e6


async def main() -> None:
    args = parse_args(__doc__.splitlines()[1])
    engine = decision_model(args)
    strategy = strategy_for(args, engine)
    shared = router(strategy)
    report = Report(args.engine, engine)
    floor = MIN_CONFIDENCE if isinstance(strategy, DecisionStrategy) else None
    print(f"mixture of models ({args.engine}, {type(strategy).__name__}, min_confidence={floor})")
    declined, routed_cost = 0, 0.0
    for prompt, expected in CASES:
        before = engine.errors
        engine.last = None
        context = _context(shared, prompt)
        candidate = await report.time(lambda context=context: strategy.select(context))
        answer = engine.last.answers["candidate"] if engine.last else None
        seen = _seen(shared, answer)
        # A decline is an outcome like any other: correct on the ambiguous case, a miss on a clear one.
        chosen = DECLINE if candidate is None else candidate.model.get_config()["model_id"]
        declined += candidate is None
        routed_cost += gen_cost(COMPLEX if candidate is None else chosen)
        report.check(prompt[:48], expected, chosen, seen, errors_before=before)
    always_complex = gen_cost(COMPLEX) * len(CASES)
    per_1k = 1000 / len(CASES)
    print(
        f"\ndeclined {declined}/{len(CASES)} (each served by the router default, {COMPLEX})."
        f"\ngeneration cost per 1k requests in this mix (~{GEN_TOKENS[0]} in / {GEN_TOKENS[1]} out tokens each):"
        f" routed ${routed_cost * per_1k:.2f} vs always-complex ${always_complex * per_1k:.2f},"
        " plus the decision cost below."
    )
    report.summary()

    # The agent never sees the decision: the router asks the strategy, then serves the chosen tier.
    prompt = CASES[0][0]
    picked = await strategy.select(_context(shared, prompt))
    served = (picked or shared.candidates[0]).model.get_config()["model_id"]
    answer = str(Agent(model=shared, callback_handler=None)(prompt)).strip()
    print(f"\nagent answer (routed to {served}): {answer[:60]}")


def _seen(shared: ModelRouter, answer) -> str:
    """The strategy asks over keys c0, c1, ... in candidate order; show them as tier names."""
    if answer is None:
        return "no confidence (ClassifierStrategy)"
    names = {f"c{i}": candidate.name for i, candidate in enumerate(shared.candidates)}
    named = {names[key]: p for key, p in answer.probabilities.items()}
    return f"confidence={answer.confidence} [{distribution(named)}]"


def _context(shared: ModelRouter, prompt: str) -> RoutingContext:
    return RoutingContext(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system_prompt=None,
        tool_specs=[],
        candidates=shared.candidates,
        invocation_state={},
    )


if __name__ == "__main__":
    asyncio.run(main())
