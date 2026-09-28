import enum
import logging
import math
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

from strands.experimental.decisions import (
    Choice,
    DecisionModel,
    DecisionResponse,
    Score,
    TemperatureFit,
    YesNo,
    fit_temperature,
)
from strands.experimental.decisions._calibration import answer_from_logits


class Dept(enum.Enum):
    BILLING = "billing"
    TECHNICAL = "technical"


class Triage(BaseModel):
    dept: Annotated[Literal["billing", "technical"], Choice("Which team?")]
    urgent: Annotated[bool, YesNo("Urgent?")]


class _LogitModel(DecisionModel):
    """A raw-logit provider whose logits per state are scripted; answers apply the configured temperature."""

    def __init__(self, logits_by_state, *, temperature=1.0, with_logits=True):
        self.config = {"model_id": "logit-1", "temperature": temperature}
        self._logits = logits_by_state
        self._with_logits = with_logits

    @property
    def calibrated(self):
        return True

    def get_config(self):
        return self.config

    def update_config(self, **config):
        self.config.update(config)

    async def _ask(self, state, questions, **kwargs):
        answers = {}
        for question_id, question in questions.items():
            answer = answer_from_logits(
                question, self._logits[state][question_id], temperature=self.config["temperature"]
            )
            if not self._with_logits:
                answer = type(answer)(
                    **{**answer.__dict__, "logits": None, "extras": {}},  # probabilities-only wire
                )
            answers[question_id] = answer
        return DecisionResponse(answers=answers)


def _overconfident_set(n=200):
    """Logits that are 3x too sharp: 70% of examples are right, but the model says ~99% every time."""
    logits, examples = {}, []
    for index in range(n):
        right = index % 10 < 7
        state = f"s{index}"
        logits[state] = {
            "dept": {"billing": 6.0, "technical": 0.0},
            "urgent": {True: 6.0, False: 0.0},
        }
        label = "billing" if right else "technical"
        examples.append((state, Triage(dept=label, urgent=right)))
    return logits, examples


@pytest.mark.asyncio
async def test_fit_softens_an_overconfident_model_and_reduces_ece():
    logits, examples = _overconfident_set()

    fit = await fit_temperature(_LogitModel(logits), Triage, examples)

    # The NLL optimum puts softmax(6/T) at the 70% accuracy: T = 6 / ln(0.7/0.3).
    assert isinstance(fit, TemperatureFit)
    assert float(fit) == pytest.approx(6 / math.log(0.7 / 0.3), rel=1e-3)
    assert fit.ece_after < 0.01 < 0.2 < fit.ece_before
    assert fit.nll_after < fit.nll_before
    assert fit.pairs == 400


@pytest.mark.asyncio
async def test_fit_on_probabilities_only_matches_fit_on_logits():
    logits, examples = _overconfident_set()

    from_logits = await fit_temperature(_LogitModel(logits), Triage, examples)
    from_probabilities = await fit_temperature(_LogitModel(logits, with_logits=False), Triage, examples)

    assert float(from_probabilities) == pytest.approx(float(from_logits), rel=1e-3)


@pytest.mark.asyncio
async def test_fit_returns_the_absolute_temperature_when_one_is_already_configured():
    logits, examples = _overconfident_set()

    configured = await fit_temperature(_LogitModel(logits, temperature=2.0, with_logits=False), Triage, examples)

    assert float(configured) == pytest.approx(6 / math.log(0.7 / 0.3), rel=1e-3)
    assert configured.ece_before > configured.ece_after


class Ticket(BaseModel):
    dept: Annotated[Dept | None, Choice("Which team?")]
    anger: Annotated[float, Score("How angry?", levels=["calm", "upset", "furious"])]


@pytest.mark.asyncio
async def test_partial_labels_enum_none_and_score_levels(caplog):
    logits = {
        "a": {"dept": {"billing": 2.0, "technical": 0.0, "none": 0.0}, "anger": {0: 0.0, 1: 2.0, 2: 0.0}},
        "b": {"dept": {"billing": 0.0, "technical": 0.0, "none": 2.0}, "anger": {0: 2.0, 1: 0.0, 2: 0.0}},
    }
    examples = [("a", {"dept": Dept.BILLING, "anger": 1}), ("b", {"dept": None})]

    with caplog.at_level(logging.WARNING):
        fit = await fit_temperature(_LogitModel(logits), Ticket, examples)

    assert fit.pairs == 3
    assert "small set" in caplog.text
    assert "TemperatureFit(" in repr(fit)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"dept": "sales"}, "not one of the options"),
        ({"urgent": 1}, "must be a bool"),
        ({"unknown": True}, "does not decide"),
    ],
)
async def test_fit_rejects_invalid_labels(expected, message):
    logits, _ = _overconfident_set(1)
    with pytest.raises(ValueError, match=message):
        await fit_temperature(_LogitModel(logits), Triage, [("s0", expected)])


@pytest.mark.asyncio
@pytest.mark.parametrize("anger", [3, 1.5, True])
async def test_fit_rejects_invalid_score_labels(anger):
    with pytest.raises(ValueError, match="level index"):
        await fit_temperature(_LogitModel({}), Ticket, [("a", {"anger": anger})])


@pytest.mark.asyncio
async def test_fit_rejects_empty_examples_bad_concurrency_and_wrong_instance():
    with pytest.raises(ValueError, match="at least one"):
        await fit_temperature(_LogitModel({}), Triage, [])
    with pytest.raises(ValueError, match="max_concurrency"):
        await fit_temperature(_LogitModel({}), Triage, [("s", {})], max_concurrency=0)
    with pytest.raises(ValueError, match="expected must be a Triage"):
        await fit_temperature(_LogitModel({}), Triage, [("s", Ticket(dept=None, anger=0.0))])
