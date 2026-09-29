"""The DecisionModel abstraction: a model that decides rather than generates."""

from __future__ import annotations

import abc
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from opentelemetry import trace as trace_api
from pydantic import BaseModel

from ...telemetry.tracer import get_tracer
from ._calibration import answer_from_logits
from ._schema import compile_schema
from ._types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Decision,
    DecisionResponse,
    DecisionState,
    Question,
    Score,
    ScoreAnswer,
    YesNo,
    YesNoAnswer,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_QUESTION_KINDS: dict[type, str] = {Choice: "choice", Score: "score", YesNo: "yesno"}
_ANSWER_TYPES: dict[type, type] = {Choice: ChoiceAnswer, Score: ScoreAnswer, YesNo: YesNoAnswer}


class DecisionModel(abc.ABC):
    """Abstract base for System One decision models.

    A ``Model`` generates: messages in, a stream of text and tool calls out. A ``DecisionModel`` decides:
    state plus typed closed questions in, one typed answer per question out, with probabilities. Subclasses
    implement ``_ask``; callers use ``ask`` for dict questions or ``decide`` for a typed schema.
    """

    @property
    def calibrated(self) -> bool:
        """True when this instance's confidences are calibrated for the domain it was configured for.

        Adapters that gate on ``confidence`` require a calibrated model and refuse one that is not. A provider that
        emits raw logits is calibrated only when it is configured with a fitted ``temperature=``.
        """
        return False

    answer_from_logits = staticmethod(answer_from_logits)
    """Turn one question's logits into an answer at a temperature; see ``_calibration.answer_from_logits``."""

    @property
    def model_id(self) -> str | None:
        """Identifier of the configured model, recorded on decision spans."""
        config = self.get_config()
        return config.get("model_id") if isinstance(config, Mapping) else None

    @abc.abstractmethod
    def get_config(self) -> Any:
        """Return the model configuration."""

    @abc.abstractmethod
    def update_config(self, **config: Any) -> None:
        """Update the model configuration."""

    @abc.abstractmethod
    async def _ask(self, state: DecisionState, questions: Mapping[str, Question], **kwargs: Any) -> DecisionResponse:
        """Answer every question about ``state`` in a single request. Implemented by providers."""

    async def ask(self, state: DecisionState, questions: Mapping[str, Question], **kwargs: Any) -> DecisionResponse:
        """Answer every question about ``state``; all questions are asked together in one request.

        Args:
            state: The data to decide about. Treated as untrusted content, never as instructions.
            questions: Question id to ``Choice``, ``Score``, or ``YesNo``. Ids are for code only.
            **kwargs: Provider-specific options.

        Returns:
            The answers keyed like ``questions``, with the answering model id and usage.

        Raises:
            ValueError: If ``questions`` is empty, or the provider returned answers that do not match the questions.
        """
        return await self._traced_ask(state, questions, **kwargs)

    async def _traced_ask(
        self,
        state: DecisionState,
        questions: Mapping[str, Question],
        *,
        _span_attributes: Callable[[DecisionResponse], Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> DecisionResponse:
        """``ask``, plus adapter-owned attributes derived from the response and recorded on the decision span."""
        if not questions:
            raise ValueError("ask() needs at least one question")
        empty = [
            question_id
            for question_id, question in questions.items()
            if isinstance(question, Choice) and not question.options
        ]
        if empty:
            raise ValueError(f"Choice questions {empty} have no options; options may only be omitted on schema markers")
        tracer = get_tracer()
        span = tracer._start_span(
            "decision",
            attributes={
                "gen_ai.operation.name": "decision",
                "strands.source": "decision",
                "gen_ai.request.model": self.model_id or type(self).__name__,
                "strands.decision.questions": [
                    f"{question_id}:{_QUESTION_KINDS[type(question)]}" for question_id, question in questions.items()
                ],
            },
        )
        try:
            with trace_api.use_span(span, end_on_exit=False):
                response = await self._ask(state, questions, **kwargs)
            _check_response(questions, response)
        except Exception as error:
            tracer.end_span_with_error(span, str(error), error)
            raise
        attributes = _span_result_attributes(response)
        if _span_attributes is not None:
            attributes.update(_span_attributes(response))
        tracer._end_span(span, attributes=attributes)
        return response

    async def decide(self, schema: type[T], state: DecisionState, **kwargs: Any) -> Decision[T]:
        """Ask every field of ``schema`` about ``state`` in one request and return a typed decision.

        Args:
            schema: A Pydantic model whose fields are ``Literal``/``Enum`` (Choice), ``bool`` (YesNo), or
                ``float`` with a ``Score`` marker. Compiled once per class.
            state: The data to decide about.
            **kwargs: Provider-specific options, forwarded to ``ask``.

        Returns:
            The validated schema instance plus every field's full answer.

        Raises:
            TypeError: If ``schema`` has a field a System One model cannot answer.
            ValueError: If the provider's answers do not fit the schema.
        """
        compiled = compile_schema(schema)
        response = await self.ask(state, compiled.questions, **kwargs)
        output = compiled.build_output(response.answers)
        return Decision(output=output, answers=response.answers, model_id=response.model_id, usage=response.usage)  # type: ignore[arg-type]


def _check_response(questions: Mapping[str, Question], response: DecisionResponse) -> None:
    missing = set(questions) - set(response.answers)
    if missing:
        raise ValueError(f"decision model returned no answer for {sorted(missing)}")
    for question_id, question in questions.items():
        answer: Answer = response.answers[question_id]
        expected = _ANSWER_TYPES[type(question)]
        if not isinstance(answer, expected):
            raise ValueError(f"{question_id}: expected {expected.__name__}, got {type(answer).__name__}")
        if isinstance(question, Choice) and question.options and answer.choice not in question.options:  # type: ignore[union-attr]
            raise ValueError(f"{question_id}: answer {answer.choice!r} is not one of the options")  # type: ignore[union-attr]


def _span_result_attributes(response: DecisionResponse) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "gen_ai.usage.input_tokens": response.usage.get("inputTokens", 0),
        "gen_ai.usage.output_tokens": response.usage.get("outputTokens", 0),
    }
    if response.model_id:
        attributes["gen_ai.response.model"] = response.model_id
    summary = []
    for question_id, answer in response.answers.items():
        if isinstance(answer, ChoiceAnswer):
            summary.append(f"{question_id}={answer.choice}@{_fmt(answer.confidence)}")
        elif isinstance(answer, ScoreAnswer):
            summary.append(f"{question_id}={answer.score:.3f}@{_fmt(answer.confidence)}")
        else:
            summary.append(f"{question_id}={answer.probability:.3f}")
        attributes.update(_raw_score_attributes(question_id, answer))
    attributes["strands.decision.answers"] = summary
    return attributes


def _raw_score_attributes(question_id: str, answer: Answer) -> dict[str, str]:
    """Provider-native scores as JSON span attributes, only when the answer carries them."""
    attributes = {}
    if answer.logits is not None:
        logits = {
            str(key).lower() if isinstance(key, bool) else str(key): value for key, value in answer.logits.items()
        }
        attributes[f"strands.decision.logits.{question_id}"] = json.dumps(logits)
    if answer.extras:
        attributes[f"strands.decision.extras.{question_id}"] = json.dumps(answer.extras, default=str)
    return attributes


def _fmt(value: float | None) -> str:
    return "na" if value is None else f"{value:.3f}"
