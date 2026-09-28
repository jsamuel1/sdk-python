"""Fit one decision-model temperature offline from labelled examples.

Fitting is a one-off step: run it on 100-200 labelled ``(state, expected)`` pairs, then paste the returned number into
the provider's ``temperature=``. A model never fits itself at runtime.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ._calibration import logits_of, softmax
from ._model import DecisionModel
from ._schema import NO_MATCH_OPTION, compile_schema
from ._types import Answer, Choice, ChoiceAnswer, DecisionState, Question, Score, ScoreAnswer, YesNo, YesNoAnswer

logger = logging.getLogger(__name__)

_MIN_TEMPERATURE = 0.05
_MAX_TEMPERATURE = 20.0
_SEARCH_STEPS = 80
_ECE_BINS = 10
_RECOMMENDED_EXAMPLES = 100

Example = tuple[DecisionState, "BaseModel | Mapping[str, Any]"]


class TemperatureFit(float):
    """The fitted temperature, as a float, with the calibration it achieved on the fitting set.

    Pass it straight to a provider's ``temperature=``. ``ece_before`` is the expected calibration error of the model
    as configured, ``ece_after`` at the fitted temperature; both use top-label confidence in 10 equal-width bins over
    every scored (example, question) pair.
    """

    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    pairs: int

    def __new__(cls, temperature: float, **report: float) -> TemperatureFit:
        """Create the fit result; ``report`` holds the before/after metrics and the scored pair count."""
        fit = super().__new__(cls, temperature)
        for name, value in report.items():
            setattr(fit, name, value)
        return fit

    def __repr__(self) -> str:
        """Show the temperature and its calibration report."""
        return (
            f"TemperatureFit({float(self):.4f}, ece {self.ece_before:.4f} -> {self.ece_after:.4f}, "
            f"nll {self.nll_before:.4f} -> {self.nll_after:.4f}, pairs={self.pairs})"
        )


@dataclass(frozen=True)
class _Pair:
    """One scored question on one example: base logits and the index-free label key."""

    logits: Mapping[Any, float]
    label: Any


async def fit_temperature(
    model: DecisionModel,
    schema: type[BaseModel],
    examples: Sequence[Example],
    *,
    max_concurrency: int = 8,
) -> TemperatureFit:
    """Fit one scalar temperature for ``model`` on ``schema`` by minimising negative log-likelihood.

    Each example is ``(state, expected)``: ``expected`` is a ``schema`` instance, or a mapping of field name to the
    expected value when only some fields are labelled. An optional Choice's no-match is labelled ``None``; a Score
    field is labelled with its level index. The fit uses the answer's ``logits`` when the provider returns them, and
    otherwise the log of its probabilities (``softmax(log p / T)`` equals rescaling the logits), scaled by the
    configured ``temperature`` so the result is always the absolute value to configure.

    Args:
        model: The decision model to fit, as it will be configured apart from its temperature.
        schema: The decision schema asked of every example.
        examples: Labelled ``(state, expected)`` pairs; about 100-200 is enough.
        max_concurrency: Maximum requests in flight.

    Returns:
        The fitted temperature, a float carrying ``ece_before``/``ece_after`` and ``nll_before``/``nll_after``.

    Raises:
        TypeError: If ``schema`` cannot be asked of a decision model.
        ValueError: If there are no examples, an expected value is not a valid answer for its field, or
            ``max_concurrency`` is not positive.
    """
    if not examples:
        raise ValueError("fit_temperature needs at least one labelled example")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")
    compiled = compile_schema(schema)
    questions = compiled.questions
    base_temperature = _configured_temperature(model)
    labels = [_labels(schema, questions, expected) for _, expected in examples]

    semaphore = asyncio.Semaphore(max_concurrency)

    async def ask(state: DecisionState) -> Mapping[str, Answer]:
        async with semaphore:
            return (await model.ask(state, questions)).answers

    responses = await asyncio.gather(*(ask(state) for state, _ in examples))
    pairs = [
        pair
        for answers, labelled in zip(responses, labels, strict=True)
        for pair in _pairs(answers, labelled, base_temperature)
    ]
    if len(examples) < _RECOMMENDED_EXAMPLES:
        logger.warning(
            "examples=<%d>, recommended=<%d> | temperature fitted on a small set; expect a noisy estimate",
            len(examples),
            _RECOMMENDED_EXAMPLES,
        )

    before = base_temperature
    fitted = _minimise_nll(pairs)
    fit = TemperatureFit(
        fitted,
        ece_before=_ece(pairs, before),
        ece_after=_ece(pairs, fitted),
        nll_before=_nll(pairs, before),
        nll_after=_nll(pairs, fitted),
        pairs=len(pairs),
    )
    logger.info("fit=<%r> | temperature fitted", fit)
    return fit


def _configured_temperature(model: DecisionModel) -> float:
    config = model.get_config()
    temperature = config.get("temperature") if isinstance(config, Mapping) else None
    return float(temperature) if isinstance(temperature, (int, float)) and not isinstance(temperature, bool) else 1.0


def _labels(
    schema: type[BaseModel], questions: Mapping[str, Question], expected: BaseModel | Mapping[str, Any]
) -> dict[str, Any]:
    values = expected.model_dump() if isinstance(expected, BaseModel) else dict(expected)
    if isinstance(expected, BaseModel) and not isinstance(expected, schema):
        raise ValueError(f"expected must be a {schema.__name__} or a mapping, got {type(expected).__name__}")
    unknown = set(values) - set(questions)
    if unknown:
        raise ValueError(f"expected has fields {sorted(unknown)} that {schema.__name__} does not decide")
    return {name: _label(name, questions[name], value) for name, value in values.items()}


def _label(name: str, question: Question, value: Any) -> Any:
    if isinstance(question, YesNo):
        return _yes_no_label(name, value)
    if isinstance(question, Choice):
        return _choice_label(name, question, value)
    assert isinstance(question, Score)
    return _score_label(name, question, value)


def _yes_no_label(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name}: a yes/no label must be a bool, got {value!r}")
    return value


def _choice_label(name: str, question: Choice, value: Any) -> str:
    if value is None:
        option = NO_MATCH_OPTION
    else:
        option = value.value if isinstance(value, enum.Enum) else value
    if option not in question.options:
        raise ValueError(f"{name}: label {value!r} is not one of the options")
    return str(option)


def _score_label(name: str, question: Score, value: Any) -> int:
    levels = len(question.levels)
    is_index = isinstance(value, int) and not isinstance(value, bool) and 0 <= value < levels
    if not is_index:
        raise ValueError(f"{name}: a score label must be a level index from 0 to {levels - 1}, got {value!r}")
    return int(value)


def _pairs(answers: Mapping[str, Answer], labels: Mapping[str, Any], base_temperature: float) -> list[_Pair]:
    return [_Pair(_base_logits(answers[name], base_temperature), label) for name, label in labels.items()]


def _base_logits(answer: Answer, base_temperature: float) -> dict[Any, float]:
    """Pre-temperature scores: the provider's logits, or log-probabilities undone by the configured temperature."""
    if answer.logits is not None:
        return dict(answer.logits)
    if isinstance(answer, YesNoAnswer):
        probabilities: Mapping[Any, float] = {True: answer.probability, False: 1.0 - answer.probability}
    else:
        assert isinstance(answer, (ChoiceAnswer, ScoreAnswer))
        probabilities = answer.probabilities
    return {key: value * base_temperature for key, value in logits_of(probabilities).items()}


def _nll(pairs: Sequence[_Pair], temperature: float) -> float:
    total = 0.0
    for pair in pairs:
        scaled = {key: value / temperature for key, value in pair.logits.items()}
        top = max(scaled.values())
        log_total = top + math.log(sum(math.exp(value - top) for value in scaled.values()))
        total += log_total - scaled[pair.label]
    return total / len(pairs)


def _minimise_nll(pairs: Sequence[_Pair]) -> float:
    """Golden-section search on beta = 1/T, where the temperature-scaling NLL is convex."""
    low, high = 1.0 / _MAX_TEMPERATURE, 1.0 / _MIN_TEMPERATURE
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    left_nll, right_nll = _nll(pairs, 1 / left), _nll(pairs, 1 / right)
    for _ in range(_SEARCH_STEPS):
        if left_nll < right_nll:
            high, right, right_nll = right, left, left_nll
            left = high - ratio * (high - low)
            left_nll = _nll(pairs, 1 / left)
        else:
            low, left, left_nll = left, right, right_nll
            right = low + ratio * (high - low)
            right_nll = _nll(pairs, 1 / right)
    return 1 / ((low + high) / 2)


def _ece(pairs: Sequence[_Pair], temperature: float) -> float:
    """Top-label expected calibration error over equal-width confidence bins."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(_ECE_BINS)]
    for pair in pairs:
        probabilities = softmax(pair.logits, temperature)
        predicted = max(probabilities, key=lambda key: probabilities[key])
        confidence = probabilities[predicted]
        bins[min(int(confidence * _ECE_BINS), _ECE_BINS - 1)].append((confidence, predicted == pair.label))
    return sum(
        len(members)
        / len(pairs)
        * abs(sum(c for c, _ in members) / len(members) - sum(ok for _, ok in members) / len(members))
        for members in bins
        if members
    )
