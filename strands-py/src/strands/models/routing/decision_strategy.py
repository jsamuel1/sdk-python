"""Route among configured candidates by asking a decision model."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ...decisions import DecisionModel, Question, Uncertain
from ...types.content import ContentBlock, Message, Messages, SystemPrompt
from .router import RoutingCandidate
from .strategy import RoutingContext

logger = logging.getLogger(__name__)

_DEFAULT_MESSAGE_CHARACTER_LIMIT = 4_000
_DEFAULT_AGENT_INSTRUCTIONS_CHARACTER_LIMIT = 4_000
_DEFAULT_CANDIDATE_CHARACTER_LIMIT = 4_000
_DEFAULT_TIMEOUT = 30.0
_ROUTING_OMISSION_MARKER = "\n...[content omitted for routing]...\n"
_NO_REQUEST_TEXT = "[No request-bearing user message provided]"
_DEFAULT_QUESTION = (
    "Which candidate, identified by its key, should serve the latest request? Judge only the latest request, "
    "agent instructions, and candidate evidence in the routing context."
)
_QUESTION_ID = "candidate"


@dataclass(frozen=True)
class _DeclineLog:
    """Log reasons and message a strategy reports when it declines selection.

    Attributes:
        error: Reason for a decision call that raised.
        timeout: Reason for a decision call that exceeded ``timeout``.
        uncertain: Reason for an ``Uncertain`` answer.
        unknown_key: Reason for an answer that matches no candidate key.
        message: Message suffix after the structured fields.
    """

    error: str
    timeout: str
    uncertain: str
    unknown_key: str
    message: str


_DECISION_DECLINE_LOG = _DeclineLog(
    error="decision_error",
    timeout="decision_timeout",
    uncertain="decision_uncertain",
    unknown_key="unknown_candidate_key",
    message="selection declined",
)


class DecisionStrategy:
    """Choose a candidate by asking a ``DecisionModel`` one closed question over the candidates.

    Each candidate is offered under a positional key (``c0``, ``c1``, ...); names, descriptions, and metadata travel
    only as evidence in the routing context, so optional or arbitrary names never become answer values. The routing
    context is a bounded JSON projection of the latest request, the parent agent's textual instructions, and candidate
    evidence; it crosses the decision model's boundary and must not contain secrets.

    Selection runs only before the first model call. A single candidate is returned without asking. The routing
    question sets ``uncertainty`` to ``False``, so the decision model must commit to a key. Decision failures, unknown
    keys, ``Uncertain`` answers from implementations that ignore that setting, and timeouts warn and decline selection,
    so ``ModelRouter`` serves candidate zero. After an attempt this strategy declines, so a failure of the selected
    candidate surfaces without switching.

    The decision's reported token usage is added to the request in ``RoutingContext.invocation_context``; the
    request's limits do not gate routing.

    Example:
        ```python
        router = ModelRouter(
            [fast, strong],
            strategy=DecisionStrategy(LLMDecisionModel(classifier_model)),
        )
        agent = Agent(model=router)
        ```
    """

    _decline_log: ClassVar[_DeclineLog] = _DECISION_DECLINE_LOG
    """Log vocabulary used when declining; presets override it to keep their own log contract."""

    def __init__(
        self,
        decision_model: DecisionModel,
        *,
        question: str = _DEFAULT_QUESTION,
        timeout: float = _DEFAULT_TIMEOUT,
        max_message_chars: int = _DEFAULT_MESSAGE_CHARACTER_LIMIT,
        max_agent_instructions_chars: int = _DEFAULT_AGENT_INSTRUCTIONS_CHARACTER_LIMIT,
        max_candidate_chars: int = _DEFAULT_CANDIDATE_CHARACTER_LIMIT,
    ) -> None:
        """Initialize the strategy.

        Args:
            decision_model: Decision model asked to choose among candidate keys.
            question: Question posed to the decision model. Defaults to an SDK routing question over candidate keys.
            timeout: Maximum seconds to wait for the decision. A timed-out decision is cancelled. The timeout runs
                independently of agent cancellation: ``RoutingContext`` carries no cancel signal, so cancelling the
                agent while a decision is in flight does not cancel that decision.
            max_message_chars: Maximum characters copied from the latest request into the routing context.
            max_agent_instructions_chars: Maximum characters copied from the parent agent's system prompt text into
                the routing context.
            max_candidate_chars: Maximum aggregate characters for the serialized evidence (keys, names, descriptions,
                and metadata) of all candidates. Evidence is never truncated; selection raises ``ValueError`` when
                the budget is exceeded.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``question`` is empty, ``timeout`` is not finite and greater than zero, or a character
                limit is not positive.
        """
        if not isinstance(decision_model, DecisionModel):
            raise TypeError("decision_model must be a DecisionModel")
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must be a non-empty string")

        self._decision_model = decision_model
        self._question = question
        self._timeout = _validate_timeout(timeout)
        self._max_message_chars = _validate_character_limit("max_message_chars", max_message_chars)
        self._max_agent_instructions_chars = _validate_character_limit(
            "max_agent_instructions_chars", max_agent_instructions_chars
        )
        self._max_candidate_chars = _validate_character_limit("max_candidate_chars", max_candidate_chars)

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate | None:
        """Select one opening candidate, declining on a decision failure.

        Args:
            context: Current request and chronological routing history.
            **kwargs: Reserved for future extensions of the routing protocol.

        Returns:
            The chosen candidate from ``context.candidates``, or ``None`` to decline.

        Raises:
            ValueError: If the candidates' serialized evidence exceeds ``max_candidate_chars``. This
                misconfiguration is permanent, so it propagates instead of declining.
        """
        if context.attempts:
            return None
        if len(context.candidates) == 1:
            return context.candidates[0]

        profiles = _build_candidate_profiles(context.candidates, self._max_candidate_chars)
        try:
            # wait_for cancels the decision task on timeout, so the in-flight call is aborted, not abandoned.
            answer = await asyncio.wait_for(self._decide(context, profiles), timeout=self._timeout)
        except asyncio.TimeoutError as error:
            self._warn_declined(self._decline_log.timeout, type(error).__name__)
            return None
        except Exception as error:
            # Logs only the error type: decision failures may carry request or candidate evidence.
            self._warn_declined(self._decline_log.error, type(error).__name__)
            return None

        if isinstance(answer, Uncertain):
            self._warn_declined(self._decline_log.uncertain)
            return None
        index = next((index for index, profile in enumerate(profiles) if profile["key"] == answer), None)
        if index is None:
            self._warn_declined(self._decline_log.unknown_key)
            return None
        return context.candidates[index]

    async def _decide(self, context: RoutingContext, profiles: Sequence[dict[str, Any]]) -> bool | str | Uncertain:
        """Ask the decision model for one candidate key over the bounded routing context."""
        state = json.dumps(
            {
                "latest_request": _latest_request_text(context.messages, self._max_message_chars),
                "agent_instructions": _extract_bounded_agent_instructions(
                    context.system_prompt, self._max_agent_instructions_chars
                ),
                "candidates": list(profiles),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        question: Question = {
            "question": self._question,
            "choices": [profile["key"] for profile in profiles],
            "uncertainty": False,
        }
        # ``ask`` adds the decision's reported usage to the request; the request's limits don't gate routing.
        result = await self._decision_model.ask(
            state, {_QUESTION_ID: question}, invocation_context=context.invocation_context
        )
        return result.answers[_QUESTION_ID]

    def _warn_declined(self, reason: str, error_type: str | None = None) -> None:
        """Log a decline using only the reason and, for failures, the error type."""
        error_field = "" if error_type is None else f", error_type=<{error_type}>"
        logger.warning(
            "strategy=<%s>, reason=<%s>%s | %s",
            type(self).__name__,
            reason,
            error_field,
            self._decline_log.message,
        )


def _candidate_key(index: int) -> str:
    """Return the positional key offered to the decision model for the candidate at ``index``."""
    return f"c{index}"


def _build_candidate_profiles(
    candidates: Sequence[RoutingCandidate],
    character_limit: int,
) -> tuple[dict[str, Any], ...]:
    """Build keyed candidate profiles, rejecting evidence that exceeds the aggregate budget.

    Candidate evidence is caller-authored configuration, so it is never truncated; an over-budget profile set
    raises instead of silently degrading the decision basis.

    Raises:
        ValueError: If the serialized profiles exceed ``character_limit``.
    """
    profiles: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        evidence = {
            "name": candidate.name,
            "description": candidate.description,
            "metadata": candidate.metadata,
        }
        profiles.append(
            {"key": _candidate_key(index), **{key: value for key, value in evidence.items() if value is not None}}
        )
    serialized_size = len(json.dumps(profiles, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    if serialized_size > character_limit:
        raise ValueError(
            f"candidate evidence serializes to {serialized_size} characters, exceeding max_candidate_chars="
            f"{character_limit}; trim candidate names, descriptions, and metadata, or raise the limit"
        )
    return tuple(profiles)


def _validate_timeout(timeout: object) -> float:
    """Return a validated finite, positive timeout in seconds."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a number")
    value = float(timeout)
    if not 0 < value < float("inf"):
        raise ValueError("timeout must be finite and greater than zero")
    return value


def _validate_character_limit(name: str, value: object) -> int:
    """Return a validated positive character limit."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _truncate_text(text: str, character_limit: int) -> str:
    """Bound text while preserving its opening and trailing request."""
    if len(text) <= character_limit:
        return text
    if character_limit <= len(_ROUTING_OMISSION_MARKER):
        return text[:character_limit]
    available_characters = character_limit - len(_ROUTING_OMISSION_MARKER)
    head_characters = available_characters // 2
    tail_characters = available_characters - head_characters
    return f"{text[:head_characters]}{_ROUTING_OMISSION_MARKER}{text[-tail_characters:]}"


def _project_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _project_guarded(value: object) -> str | None:
    """Report guarded text as present without forwarding it."""
    text = value.get("text") if isinstance(value, Mapping) else None
    guarded = text.get("text") if isinstance(text, Mapping) else None
    return "[Guarded content]" if isinstance(guarded, str) and guarded.strip() else None


def _label(label: str) -> Callable[[object], str]:
    return lambda _value: label


def _not_request_bearing(_value: object) -> None:
    return None


_REQUEST_BLOCK_PROJECTIONS: Mapping[str, Callable[[object], str | None]] = {
    "text": _project_text,
    "guardContent": _project_guarded,
    "audio": _label("[Audio]"),
    "image": _label("[Image]"),
    "document": _label("[Document]"),
    "video": _label("[Video]"),
    "toolUse": _not_request_bearing,
    "toolResult": _not_request_bearing,
    "reasoningContent": _not_request_bearing,
    "cachePoint": _not_request_bearing,
    "citationsContent": _not_request_bearing,
}
"""Safe routing projection for every ``ContentBlock`` key.

Opaque payloads (tool results, reasoning, guarded text) never reach the routing context; media is reported by
label only. A test pins these keys to ``ContentBlock``'s, so a new block type cannot silently fall back to stale
history.
"""


def _project_request_block(block: ContentBlock) -> list[str]:
    """Project each key a block carries through its entry in ``_REQUEST_BLOCK_PROJECTIONS``."""
    parts: list[str] = []
    for key, value in block.items():
        project = _REQUEST_BLOCK_PROJECTIONS.get(key, _not_request_bearing)
        if (part := project(value)) is not None:
            parts.append(part)
    return parts


def _request_text(message: Message, character_limit: int) -> str | None:
    """Render only safe request-bearing fields from one user message."""
    parts = [part for block in message["content"] for part in _project_request_block(block)]
    if not parts:
        return None
    return _truncate_text("\n".join(parts), character_limit)


def _latest_request_text(
    messages: Messages,
    character_limit: int = _DEFAULT_MESSAGE_CHARACTER_LIMIT,
) -> str:
    """Return the latest request-bearing user message as bounded safe text."""
    for message in reversed(messages):
        if message["role"] == "user" and (request_text := _request_text(message, character_limit)) is not None:
            return request_text
    return _truncate_text(_NO_REQUEST_TEXT, character_limit)


def _extract_bounded_agent_instructions(system_prompt: SystemPrompt | None, character_limit: int) -> str:
    """Extract bounded text from the parent agent system prompt, omitting non-text blocks."""
    if isinstance(system_prompt, str):
        instructions = system_prompt
    elif system_prompt:
        instructions = "\n".join(block["text"] for block in system_prompt if "text" in block)
    else:
        instructions = ""
    return _truncate_text(instructions, character_limit)
