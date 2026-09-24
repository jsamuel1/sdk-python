"""Temperature arithmetic shared by every provider: logits (or log-probabilities) to calibrated answers.

A provider returns scores; this module turns them into ``probabilities``, ``score`` and ``confidence`` at a
temperature, so no vendor reimplements scaling. Rescaling log-probabilities, ``softmax(log p / T)``, is the same as
rescaling the logits they came from, so a provider whose wire returns only probabilities uses ``logits_of`` first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeVar

from ._types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Question,
    Score,
    ScoreAnswer,
    YesNoAnswer,
    max_probability_confidence,
    yes_no_confidence,
)

K = TypeVar("K")

_LOG_FLOOR = 1e-12  # log(0) is -inf; floor zero probabilities so a rescale stays finite


def check_temperature(temperature: float) -> float:
    """Return ``temperature`` if it is a finite number above zero.

    Raises:
        ValueError: If it is not.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(f"temperature must be a finite number above 0, got {temperature!r}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a finite number above 0, got {temperature!r}")
    return float(temperature)


def logits_of(probabilities: Mapping[K, float]) -> dict[K, float]:
    """Log-probabilities, usable as logits: softmax recovers the (normalized) distribution."""
    return {key: math.log(max(value, _LOG_FLOOR)) for key, value in probabilities.items()}


def softmax(logits: Mapping[K, float], temperature: float = 1.0) -> dict[K, float]:
    """``softmax(logits / temperature)``, computed stably."""
    scaled = {key: value / temperature for key, value in logits.items()}
    top = max(scaled.values())
    weights = {key: math.exp(value - top) for key, value in scaled.items()}
    total = sum(weights.values())
    return {key: weight / total for key, weight in weights.items()}


def answer_from_logits(
    question: Question,
    logits: Mapping[Any, float],
    *,
    temperature: float = 1.0,
    calibrated: bool = True,
    extras: Mapping[str, Any] | None = None,
) -> Answer:
    """Build an answer from per-option (Choice), per-level (Score) or ``True``/``False`` (YesNo) logits.

    Probabilities are ``softmax(logits / temperature)``. Confidence uses the default derivation:
    ``max_probability_confidence`` for Choice and Score, ``yes_no_confidence`` for YesNo, or None when the
    instance is not calibrated. ``logits`` and ``extras`` are kept on the answer unchanged.

    Raises:
        ValueError: If ``temperature`` is not a finite number above 0, or the logits do not match the question.
    """
    check_temperature(temperature)
    if not logits:
        raise ValueError("answer_from_logits needs at least one logit")
    probabilities = softmax(logits, temperature)
    kept = dict(extras or {})
    if isinstance(question, Choice):
        choice = max(probabilities, key=lambda key: probabilities[key])
        confidence = max_probability_confidence(probabilities) if calibrated else None
        return ChoiceAnswer(choice, probabilities, confidence, logits=dict(logits), extras=kept)
    if isinstance(question, Score):
        score = sum(level * p for level, p in probabilities.items())
        confidence = max_probability_confidence(probabilities) if calibrated else None
        return ScoreAnswer(score, probabilities, confidence, logits=dict(logits), extras=kept)
    probability = probabilities[True]
    confidence = yes_no_confidence(probability) if calibrated else None
    return YesNoAnswer(probability, confidence, logits=dict(logits), extras=kept)
