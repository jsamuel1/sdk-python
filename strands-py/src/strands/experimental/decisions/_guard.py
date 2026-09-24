"""Guardrails that gate tool calls on calibrated System One judgments."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ...hooks.events import AfterModelCallEvent, BeforeInvocationEvent, BeforeToolCallEvent
from ...interventions.actions import Confirm, Deny, Guide, Proceed
from ...interventions.handler import InterventionHandler
from ...vended_interventions.hitl.classifier import ClassifierResult, HumanInTheLoopClassifier
from ._model import DecisionModel
from ._types import Question, Score, ScoreAnswer, YesNo, YesNoAnswer

logger = logging.getLogger(__name__)

GuardOnError = Literal["confirm", "deny", "proceed"]

_RISK_INSTRUCTIONS = (
    "Does the tool call in `tool` with arguments `input` need human approval before it runs because it is "
    "destructive or irreversible, changes production or shared state, touches credentials, personal or "
    "financial data, communicates externally, or has a large blast radius?"
)
_RISK_TRUE = "Could cause harm that is hard to undo or reaches beyond the task"
_RISK_FALSE = "Read-only, local, easily reversible, or scoped to a single non-critical resource"
_SEVERITY_LEVELS = (
    "harmless",
    "minor and easily reversible",
    "significant or hard to reverse",
    "severe, irreversible, or wide blast radius",
)


GuardAction = type[Proceed] | type[Confirm] | type[Deny]


@dataclass(frozen=True)
class GuardPolicy:
    """Code-owned floors that map one tool call's risk judgment to Proceed, Confirm, or Deny.

    At or above ``deny_above`` (and, when ``deny_min_severity`` is set, with severity at or above it) the call is
    denied. Otherwise, at or above ``confirm_above`` a person confirms. Otherwise it proceeds. A Deny blocked by
    the severity gate falls through to Confirm, the fail-safe direction.

    Attributes:
        confirm_above: Approval probability at or above which a person must confirm.
        deny_above: Approval probability at or above which the call is denied outright; None never auto-denies.
        deny_min_severity: When set, a Deny also needs the severity score (level index) at or above this.
    """

    confirm_above: float = 0.5
    deny_above: float | None = 0.95
    deny_min_severity: float | None = None

    def __post_init__(self) -> None:
        """Validate the floors.

        Raises:
            ValueError: If the floors are not ``0 <= confirm_above <= deny_above <= 1`` (or ``confirm_above`` is
                outside [0, 1] when ``deny_above`` is None), or ``deny_min_severity`` is negative.
        """
        upper = 1.0 if self.deny_above is None else self.deny_above
        if not 0.0 <= self.confirm_above <= upper <= 1.0:
            raise ValueError("GuardPolicy needs 0 <= confirm_above <= deny_above <= 1")
        if self.deny_min_severity is not None and self.deny_min_severity < 0:
            raise ValueError("GuardPolicy deny_min_severity must be >= 0")

    def action(self, probability: float, severity: float) -> GuardAction:
        """Return the action class for an approval probability and a severity score."""
        if self._denies(probability, severity):
            return Deny
        if probability >= self.confirm_above:
            return Confirm
        return Proceed

    def _denies(self, probability: float, severity: float) -> bool:
        if self.deny_above is None or probability < self.deny_above:
            return False
        return self.deny_min_severity is None or severity >= self.deny_min_severity


def _tool_state(event: BeforeToolCallEvent) -> dict[str, Any]:
    tool_use = event.tool_use
    return {"tool": tool_use["name"], "input": tool_use["input"]}


class DecisionGuard(InterventionHandler):
    """Proceed, confirm with a person, or deny each tool call by a calibrated risk judgment.

    One request per tool call asks a ``YesNo`` (does this need approval?) and a ``Score`` (how severe?) together.
    Floors are code-owned policy, set per consequence: ``policy`` applies to every tool, and ``policies`` overrides
    it by exact tool name (see ``GuardPolicy``). It fails closed by default: a decision error means ``Confirm``.
    Low confidence needs no separate floor: an unsure risk answer is a probability near 0.5, which the default
    ``confirm_above`` (0.5) sends to a person.

    ``Confirm`` pauses the agent loop until the interrupt is answered, so the application needs a way to ask a
    person (see Human-in-the-loop). An uncalibrated model (``LLMDecisionModel``) answers with one-hot
    probabilities, so it only reaches 0 or 1: the band between the floors is unreachable, and each call gets either
    Proceed or its policy's top action (Deny, or Confirm where ``deny_above`` is None).

    Optionally, ``output_questions`` check each model response: when any ``YesNo`` is at or above
    ``output_guide_above`` the response is discarded and the model retries with guidance, at most
    ``max_output_retries`` times per invocation. An output-check error proceeds (``after_model_call`` cannot
    confirm or deny) and is logged.
    """

    name = "strands:decision-guard"

    def __init__(
        self,
        decision_model: DecisionModel,
        *,
        policy: GuardPolicy | None = None,
        policies: Mapping[str, GuardPolicy] | None = None,
        on_decision_error: GuardOnError = "confirm",
        risk_instructions: str = _RISK_INSTRUCTIONS,
        severity_levels: Sequence[str] = _SEVERITY_LEVELS,
        output_questions: Mapping[str, YesNo] | None = None,
        output_guide_above: float = 0.5,
        max_output_retries: int = 1,
    ) -> None:
        """Initialize the guard.

        Args:
            decision_model: The decision model.
            policy: Floors for every tool without its own entry in ``policies``. Defaults to ``GuardPolicy()``.
            policies: Exact tool name to its own floors.
            on_decision_error: Action when the decision itself fails.
            risk_instructions: The approval question; the tool name and input are the state.
            severity_levels: Ordered severity descriptions, least severe first.
            output_questions: ``YesNo`` checks over each model response's text (``response`` in the state), each
                phrased so that yes means the response is unacceptable.
            output_guide_above: Probability at or above which an output check triggers a guided retry.
            max_output_retries: Guided retries allowed per invocation; the response is then accepted.

        Raises:
            ValueError: If a policy's ``deny_min_severity`` is above the top severity level, ``output_guide_above`` is
                outside [0, 1], ``max_output_retries`` is negative, or an output question is not a ``YesNo``.
        """
        self._policy = policy or GuardPolicy()
        self._policies = dict(policies or {})
        self._severity_max = len(severity_levels) - 1
        for name, each in {"policy": self._policy, **self._policies}.items():
            if each.deny_min_severity is not None and each.deny_min_severity > self._severity_max:
                raise ValueError(f"{name}: deny_min_severity must be <= {self._severity_max}, the top severity level")
        if not 0.0 <= output_guide_above <= 1.0 or max_output_retries < 0:
            raise ValueError("DecisionGuard needs 0 <= output_guide_above <= 1 and max_output_retries >= 0")
        if output_questions and not all(isinstance(q, YesNo) for q in output_questions.values()):
            raise ValueError("DecisionGuard output_questions must all be YesNo")
        self._output_questions: dict[str, Question] = dict(output_questions or {})
        self._output_guide_above = output_guide_above
        self._max_output_retries = max_output_retries
        self._output_retries = 0
        self._model = decision_model
        self._on_decision_error = on_decision_error
        self._questions: dict[str, Question] = {
            "needs_approval": YesNo(risk_instructions, true=_RISK_TRUE, false=_RISK_FALSE),
            "severity": Score(
                "How severe is the worst plausible outcome of this tool call?", levels=list(severity_levels)
            ),
        }

    async def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed | Deny | Confirm:
        """Gate the tool call on the judgment."""
        tool_name = event.tool_use["name"]
        try:
            response = await self._model.ask(_tool_state(event), self._questions)
        except Exception as error:
            logger.warning(
                "tool_name=<%s>, error_type=<%s>, on_decision_error=<%s> | guard decision failed",
                tool_name,
                type(error).__name__,
                self._on_decision_error,
            )
            return self._error_action(tool_name, error)
        risk = response.answers["needs_approval"]
        severity = response.answers["severity"]
        assert isinstance(risk, YesNoAnswer) and isinstance(severity, ScoreAnswer)
        reason = f"approval p={risk.probability:.2f}, severity {severity.score:.1f}/{self._severity_max}"
        action = self._policies.get(tool_name, self._policy).action(risk.probability, severity.score)
        return _build(action, tool_name, reason)

    def before_invocation(self, event: BeforeInvocationEvent, **kwargs: Any) -> Proceed:
        """Reset the per-invocation output retry budget."""
        self._output_retries = 0
        return Proceed()

    async def after_model_call(self, event: AfterModelCallEvent, **kwargs: Any) -> Proceed | Guide:
        """Check the model's response with ``output_questions``; guide a retry when a check fails."""
        response_text = _response_text(event)
        if not self._output_questions or response_text is None:
            return Proceed()
        if self._output_retries >= self._max_output_retries:
            return Proceed(reason="output retry budget exhausted")
        try:
            response = await self._model.ask({"response": response_text}, self._output_questions)
        except Exception as error:
            logger.warning("error_type=<%s> | guard output check failed, proceeding", type(error).__name__)
            return Proceed(reason=f"output check unavailable ({type(error).__name__})")
        failed = {
            question_id: answer.probability
            for question_id, answer in response.answers.items()
            if isinstance(answer, YesNoAnswer) and answer.probability >= self._output_guide_above
        }
        if not failed:
            return Proceed()
        self._output_retries += 1
        detail = ", ".join(f"{question_id} p={p:.2f}" for question_id, p in failed.items())
        feedback = "; ".join(str(self._output_questions[question_id].instructions) for question_id in failed)
        return Guide(feedback=f"Revise your response; a check flagged it: {feedback}", reason=detail)

    def _error_action(self, tool_name: str, error: Exception) -> Proceed | Deny | Confirm:
        reason = f"risk check unavailable ({type(error).__name__})"
        return _build(_ERROR_ACTIONS[self._on_decision_error], tool_name, reason)


_ERROR_ACTIONS: dict[GuardOnError, GuardAction] = {"confirm": Confirm, "deny": Deny, "proceed": Proceed}


def _build(action: GuardAction, tool_name: str, reason: str) -> Proceed | Deny | Confirm:
    if action is Deny:
        return Deny(reason=f"Blocked {tool_name}: {reason}")
    if action is Confirm:
        return Confirm(prompt=f"Allow {tool_name}? ({reason})", reason=reason)
    return Proceed(reason=reason)


def _response_text(event: AfterModelCallEvent) -> str | None:
    if event.stop_response is None:
        return None
    texts = [block["text"] for block in event.stop_response.message["content"] if isinstance(block.get("text"), str)]
    return "\n".join(texts) if texts else None


def decision_classifier(
    decision_model: DecisionModel, *, threshold: float = 0.5, instructions: str = _RISK_INSTRUCTIONS
) -> HumanInTheLoopClassifier:
    """Build a ``HumanInTheLoop`` classifier that asks for approval at or above ``threshold``.

    Use as ``HumanInTheLoop(classifier=decision_classifier(jev))``. Decision errors propagate so
    ``HumanInTheLoop`` applies its own error policy.
    """
    question: dict[str, Question] = {"needs_approval": YesNo(instructions, true=_RISK_TRUE, false=_RISK_FALSE)}

    async def classifier(event: BeforeToolCallEvent, **kwargs: Any) -> ClassifierResult:
        response = await decision_model.ask(_tool_state(event), question)
        answer = response.answers["needs_approval"]
        assert isinstance(answer, YesNoAnswer)
        return ClassifierResult(
            requires_human_in_the_loop=answer.probability >= threshold,
            reason=f"approval p={answer.probability:.2f}",
        )

    return classifier
