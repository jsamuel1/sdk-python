import math
from unittest import mock

import pytest

from strands.experimental.decisions import (
    Choice,
    ChoiceAnswer,
    DecisionModel,
    Score,
    ScoreAnswer,
    YesNo,
    YesNoAnswer,
    max_probability_confidence,
    yes_no_confidence,
)
from strands.experimental.decisions._calibration import logits_of, softmax
from tests.fixtures.mocked_decision_model import MockedDecisionModel

CHOICE = Choice("Which team?", options={"billing": None, "technical": None})
SCORE = Score("How angry?", levels=["calm", "upset", "furious"])


def test_answers_default_to_no_raw_scores():
    for answer in (
        ChoiceAnswer("a", {"a": 1.0}),
        ScoreAnswer(0.0, {0: 1.0}),
        YesNoAnswer(0.5),
    ):
        assert answer.logits is None
        assert answer.extras == {}


def test_raw_scores_round_trip_and_answers_stay_frozen():
    answer = ChoiceAnswer("a", {"a": 0.7, "b": 0.3}, 0.7, logits={"a": 1.2, "b": 0.35}, extras={"native": 0.64})

    assert answer.logits == {"a": 1.2, "b": 0.35}
    assert answer.extras == {"native": 0.64}
    assert YesNoAnswer(0.9, logits={True: 2.2, False: 0.0}).logits == {True: 2.2, False: 0.0}
    with pytest.raises(AttributeError):
        answer.logits = None  # type: ignore[misc]


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: ChoiceAnswer("a", {"a": 1.0}, logits={"b": 0.0}), "keyed exactly like"),
        (lambda: ScoreAnswer(0.0, {0: 1.0}, logits={0: math.inf}), "finite"),
        (lambda: YesNoAnswer(0.5, logits={True: 0.0}), "keyed exactly like"),
    ],
)
def test_logits_must_match_the_distribution(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_max_probability_confidence_is_the_top_probability():
    assert max_probability_confidence({"a": 0.2, "b": 0.7, "c": 0.1}) == 0.7
    assert max_probability_confidence({0: 1.0}) == 1.0
    with pytest.raises(ValueError, match="at least one probability"):
        max_probability_confidence({})


def test_softmax_of_log_probabilities_recovers_the_distribution():
    probabilities = {"a": 0.6, "b": 0.3, "c": 0.1}

    assert softmax(logits_of(probabilities)) == pytest.approx(probabilities)


def test_answer_from_logits_choice_at_a_temperature():
    logits = {"billing": 2.0, "technical": 0.0}

    tru_answer = DecisionModel.answer_from_logits(CHOICE, logits, temperature=2.0, extras={"native": 0.9})

    exp_billing = 1 / (1 + math.exp(-1.0))  # softmax([2, 0] / 2)
    assert isinstance(tru_answer, ChoiceAnswer)
    assert tru_answer.choice == "billing"
    assert tru_answer.probabilities["billing"] == pytest.approx(exp_billing)
    assert tru_answer.confidence == pytest.approx(max_probability_confidence(tru_answer.probabilities))
    assert tru_answer.logits == logits  # pre-temperature scores are kept unchanged
    assert tru_answer.extras == {"native": 0.9}


def test_answer_from_logits_temperature_one_is_plain_softmax():
    logits = {0: 0.0, 1: 1.0, 2: 3.0}

    tru_answer = DecisionModel.answer_from_logits(SCORE, logits)

    assert isinstance(tru_answer, ScoreAnswer)
    assert tru_answer.probabilities == pytest.approx(softmax(logits))
    assert tru_answer.score == pytest.approx(sum(level * p for level, p in tru_answer.probabilities.items()))
    assert tru_answer.level == 2
    assert tru_answer.confidence == pytest.approx(max(tru_answer.probabilities.values()))


def test_answer_from_logits_yes_no_uses_the_yes_no_confidence():
    tru_answer = DecisionModel.answer_from_logits(YesNo("?"), {True: 1.5, False: 0.0}, temperature=0.5)

    exp_p = 1 / (1 + math.exp(-3.0))
    assert isinstance(tru_answer, YesNoAnswer)
    assert tru_answer.probability == pytest.approx(exp_p)
    assert tru_answer.confidence == pytest.approx(yes_no_confidence(exp_p))


def test_answer_from_logits_uncalibrated_reports_no_confidence():
    for question, logits in ((CHOICE, {"billing": 1.0, "technical": 0.0}), (SCORE, {0: 0.0, 1: 1.0, 2: 0.0})):
        assert DecisionModel.answer_from_logits(question, logits, calibrated=False).confidence is None
    assert DecisionModel.answer_from_logits(YesNo("?"), {True: 0.0, False: 0.0}, calibrated=False).confidence is None


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan, math.inf, "2", True])
def test_answer_from_logits_rejects_a_bad_temperature(temperature):
    with pytest.raises(ValueError, match="temperature must be a finite number above 0"):
        DecisionModel.answer_from_logits(CHOICE, {"billing": 1.0}, temperature=temperature)


def test_answer_from_logits_needs_logits():
    with pytest.raises(ValueError, match="at least one logit"):
        DecisionModel.answer_from_logits(CHOICE, {})


@pytest.mark.asyncio
async def test_span_carries_raw_scores_only_when_present():
    raw = ChoiceAnswer("a", {"a": 0.7, "b": 0.3}, 0.7, logits={"a": 1.2, "b": 0.35}, extras={"native": 0.64})
    plain = YesNoAnswer(0.9, 0.8)
    yes_logits = YesNoAnswer(0.9, 0.8, logits={True: 2.2, False: 0.0})
    model = MockedDecisionModel({"q": raw, "p": plain, "y": yes_logits})
    tracer = mock.Mock()

    with mock.patch("strands.experimental.decisions._model.get_tracer", return_value=tracer):
        await model.ask("s", {"q": Choice("?", options={"a": None, "b": None}), "p": YesNo("?"), "y": YesNo("?")})

    attributes = tracer._end_span.call_args.kwargs["attributes"]
    assert attributes["strands.decision.logits.q"] == '{"a": 1.2, "b": 0.35}'
    assert attributes["strands.decision.extras.q"] == '{"native": 0.64}'
    assert attributes["strands.decision.logits.y"] == '{"true": 2.2, "false": 0.0}'
    assert "strands.decision.logits.p" not in attributes
    assert "strands.decision.extras.p" not in attributes
    assert "strands.decision.extras.y" not in attributes
