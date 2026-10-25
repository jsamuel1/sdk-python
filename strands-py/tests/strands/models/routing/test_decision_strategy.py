"""Tests for decision-model-driven proactive model selection."""

import asyncio
import json
import logging
import typing
from dataclasses import replace
from typing import Any

import pytest

from strands import Agent
from strands.agent.invocation_context import _create_invocation_context
from strands.decisions import DecisionInput, DecisionMetadata, DecisionModel, DecisionResult, Question, Uncertain
from strands.models import ClassifierStrategy, DecisionStrategy, ModelRouter, RoutingAttempt, RoutingCandidate
from strands.models.routing.decision_strategy import _REQUEST_BLOCK_PROJECTIONS, _latest_request_text
from strands.models.routing.strategy import RoutingContext
from strands.types.content import ContentBlock
from strands.types.event_loop import Usage
from tests.fixtures.mocked_model_provider import MockedModelProvider


class _FakeDecisionModel(DecisionModel):
    """Records each ask and answers the ``candidate`` question with a scripted value."""

    def __init__(
        self, answer: bool | str | Uncertain | Exception, *, delay: float = 0, usage: Usage | None = None
    ) -> None:
        self.answer = answer
        self.delay = delay
        self.usage = usage
        self.states: list[DecisionInput] = []
        self.questions: list[dict[str, Question]] = []
        self.cancelled = False

    async def _ask(self, state: DecisionInput, questions: dict[str, Question], **kwargs: Any) -> DecisionResult:
        self.states.append(state)
        self.questions.append(questions)
        if self.delay:
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if isinstance(self.answer, Exception):
            raise self.answer
        return DecisionResult(
            answers={"candidate": self.answer}, metadata=DecisionMetadata(latency_ms=1), usage=self.usage
        )


def _response_model(text: str) -> MockedModelProvider:
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]}])


def _candidate(name: str) -> RoutingCandidate:
    return RoutingCandidate(_response_model(name), name=name, description=f"Model for {name} requests.")


def _context(router: ModelRouter, *, messages=None, system_prompt: Any = "Be precise", attempts=()) -> RoutingContext:
    return RoutingContext(
        messages=messages or [{"role": "user", "content": [{"text": "Plan a migration"}]}],
        system_prompt=system_prompt,
        tool_specs=[],
        candidates=router.candidates,
        invocation_state={},
        attempts=attempts,
    )


@pytest.mark.asyncio
async def test_select_maps_answered_key_to_configured_candidate_instance():
    strategy = DecisionStrategy(_FakeDecisionModel("c1"))
    router = ModelRouter(models=[_candidate("routine"), _candidate("complex")], strategy=strategy)

    tru_candidate = await strategy.select(_context(router))

    assert tru_candidate is router.candidates[1]


@pytest.mark.asyncio
async def test_select_asks_one_question_over_positional_keys_with_uncertainty_disabled():
    decision_model = _FakeDecisionModel("c0")
    strategy = DecisionStrategy(decision_model, question="Which model fits?")
    router = ModelRouter(
        models=[_candidate("a"), _response_model("unnamed"), _candidate("uncertain")],
        strategy=strategy,
    )

    await strategy.select(_context(router))

    exp_questions = [
        {"candidate": {"question": "Which model fits?", "choices": ["c0", "c1", "c2"], "uncertainty": False}}
    ]
    assert decision_model.questions == exp_questions


@pytest.mark.asyncio
async def test_select_sends_bounded_request_instructions_and_keyed_evidence_as_json_state():
    decision_model = _FakeDecisionModel("c0")
    strategy = DecisionStrategy(decision_model, max_message_chars=100, max_agent_instructions_chars=50)
    router = ModelRouter(
        models=[RoutingCandidate(_response_model("a"), name="a", metadata={"cost": "low"}), _response_model("b")],
        strategy=strategy,
    )
    context = _context(
        router,
        messages=[{"role": "user", "content": [{"text": "A" * 300}]}],
        system_prompt=[{"text": "B" * 200}, {"cachePoint": {"type": "default"}}],
    )

    await strategy.select(context)

    tru_state = json.loads(decision_model.states[0])
    tru_shape = (
        len(tru_state["latest_request"]),
        "...[content omitted for routing]..." in tru_state["latest_request"],
        len(tru_state["agent_instructions"]),
        tru_state["candidates"],
    )
    exp_shape = (100, True, 50, [{"key": "c0", "name": "a", "metadata": {"cost": "low"}}, {"key": "c1"}])
    assert tru_shape == exp_shape


@pytest.mark.asyncio
async def test_select_single_candidate_bypasses_decision_model():
    decision_model = _FakeDecisionModel("c0")
    strategy = DecisionStrategy(decision_model)
    router = ModelRouter(models=[_candidate("only")], strategy=strategy)

    tru_candidate = await strategy.select(_context(router))

    assert (tru_candidate, len(decision_model.states)) == (router.candidates[0], 0)


@pytest.mark.asyncio
async def test_select_declines_after_attempt_without_asking():
    decision_model = _FakeDecisionModel("c0")
    strategy = DecisionStrategy(decision_model)
    router = ModelRouter(models=[_candidate("a"), _candidate("b")], strategy=strategy)
    attempts = (RoutingAttempt(router.candidates[0], ValueError("down")),)

    tru_candidate = await strategy.select(_context(router, attempts=attempts))

    assert (tru_candidate, len(decision_model.states)) == (None, 0)


@pytest.mark.asyncio
async def test_select_raises_when_candidate_evidence_exceeds_budget_without_asking():
    decision_model = _FakeDecisionModel("c0")
    strategy = DecisionStrategy(decision_model, max_candidate_chars=50)
    router = ModelRouter(
        models=[RoutingCandidate(_response_model("a"), description="x" * 100), _candidate("b")],
        strategy=strategy,
    )

    with pytest.raises(ValueError, match="exceeding max_candidate_chars=50; trim candidate"):
        await strategy.select(_context(router))

    assert decision_model.states == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("c9", "unknown_candidate_key"),
        ("b", "unknown_candidate_key"),
        (True, "unknown_candidate_key"),
        (Uncertain("ambiguous"), "decision_uncertain"),
        (RuntimeError("provider-secret"), "decision_error"),
    ],
    ids=["unknown-key", "name-not-key", "boolean", "uncertain", "decision-error"],
)
async def test_select_warns_safely_and_declines(answer, reason, caplog):
    strategy = DecisionStrategy(_FakeDecisionModel(answer))
    router = ModelRouter(models=[_candidate("a"), _candidate("b")], strategy=strategy)

    with caplog.at_level(logging.WARNING):
        tru_candidate = await strategy.select(_context(router))

    assert tru_candidate is None
    assert f"strategy=<DecisionStrategy>, reason=<{reason}>" in caplog.text
    assert "selection declined" in caplog.text
    assert "provider-secret" not in caplog.text


@pytest.mark.asyncio
async def test_select_timeout_cancels_decision_and_declines(caplog):
    decision_model = _FakeDecisionModel("c1", delay=1)
    strategy = DecisionStrategy(decision_model, timeout=0.01)
    router = ModelRouter(models=[_candidate("a"), _candidate("b")], strategy=strategy)

    with caplog.at_level(logging.WARNING):
        tru_candidate = await strategy.select(_context(router))

    assert (tru_candidate, decision_model.cancelled) == (None, True)
    assert "reason=<decision_timeout>, error_type=<TimeoutError>" in caplog.text


def test_agent_serves_chosen_candidate():
    router = ModelRouter(
        models=[_candidate("routine"), _candidate("complex")],
        strategy=DecisionStrategy(_FakeDecisionModel("c1")),
    )
    agent = Agent(model=router, callback_handler=None)

    tru_result = agent("hello")

    assert tru_result.message["content"][0]["text"] == "complex"


def test_agent_decision_failure_serves_candidate_zero():
    router = ModelRouter(
        models=[_candidate("default"), _candidate("other")],
        strategy=DecisionStrategy(_FakeDecisionModel(RuntimeError("down"))),
    )
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    tru_result = agent("hello")

    assert tru_result.message["content"][0]["text"] == "default"


def test_classifier_strategy_is_a_decision_strategy_preset():
    assert isinstance(ClassifierStrategy(_response_model("classifier")), DecisionStrategy)


@pytest.mark.asyncio
async def test_select_adds_decision_usage_to_request_even_when_request_limit_is_used_up():
    usage = Usage(inputTokens=4, outputTokens=5, totalTokens=9)
    strategy = DecisionStrategy(_FakeDecisionModel("c1", usage=usage))
    router = ModelRouter(models=[_candidate("a"), _candidate("b")], strategy=strategy)
    request = _create_invocation_context({"turns": 1})
    request._usage["turns_taken"] = 5

    tru_candidate = await strategy.select(replace(_context(router), invocation_context=request))

    assert tru_candidate is router.candidates[1]
    assert request.usage == {"turns_taken": 5, "inputTokens": 4, "outputTokens": 5, "totalTokens": 9}


def test_request_projection_covers_every_content_block_key():
    assert set(_REQUEST_BLOCK_PROJECTIONS) == set(typing.get_type_hints(ContentBlock))


@pytest.mark.parametrize(
    ("block", "exp_request"),
    [
        ({"audio": {"format": "mp3", "source": {"bytes": b"audio-secret"}}}, "[Audio]"),
        ({"image": {"format": "png", "source": {"bytes": b"image-secret"}}}, "[Image]"),
        ({"guardContent": {"text": {"text": "guarded-secret"}}}, "[Guarded content]"),
    ],
    ids=["audio-only", "image-only", "guarded-only"],
)
def test_latest_request_labels_non_text_requests_without_payloads(block, exp_request):
    messages = [
        {"role": "user", "content": [{"text": "stale earlier request"}]},
        {"role": "assistant", "content": [{"text": "ok"}]},
        {"role": "user", "content": [block]},
    ]

    assert _latest_request_text(messages) == exp_request


def test_latest_request_skips_non_request_bearing_blocks():
    messages = [
        {"role": "user", "content": [{"text": "real request"}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "t", "status": "success", "content": [{"text": "tool-secret"}]}},
                {"cachePoint": {"type": "default"}},
            ],
        },
    ]

    assert _latest_request_text(messages) == "real request"


def test_constructor_rejects_non_decision_model():
    with pytest.raises(TypeError, match="decision_model must be a DecisionModel"):
        DecisionStrategy(object())


@pytest.mark.parametrize(("question", "error_type"), [(None, TypeError), ("  ", ValueError)])
def test_constructor_rejects_invalid_question(question, error_type):
    with pytest.raises(error_type, match="question must be a"):
        DecisionStrategy(_FakeDecisionModel("c0"), question=question)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_constructor_rejects_non_positive_or_non_finite_timeout(timeout):
    with pytest.raises(ValueError, match="timeout must be finite and greater than zero"):
        DecisionStrategy(_FakeDecisionModel("c0"), timeout=timeout)


@pytest.mark.parametrize("name", ["max_message_chars", "max_agent_instructions_chars", "max_candidate_chars"])
@pytest.mark.parametrize(
    ("value", "error_type", "match_suffix"),
    [("100", TypeError, "must be an integer"), (0, ValueError, "must be greater than zero")],
    ids=["string", "zero"],
)
def test_constructor_rejects_invalid_character_limits(name, value, error_type, match_suffix):
    with pytest.raises(error_type, match=f"{name} {match_suffix}"):
        DecisionStrategy(_FakeDecisionModel("c0"), **{name: value})
