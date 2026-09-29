"""System-One-first action selection: a decision model serves closed-set steps before the LLM is asked."""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..._middleware.stages import InvokeModelContext, InvokeModelStage
from ...event_loop._synthesized import DECISION_SOURCE, mark_synthesized, synthesized_source
from ...models._request_text import CHARS_PER_TOKEN, project_state, truncate_text
from ...plugins import Plugin
from ...types._events import ModelStopReason
from ...types.content import Message, Messages
from ._model import DecisionModel
from ._strategy import _require_calibrated
from ._types import Choice, ChoiceAnswer, DecisionResponse

if TYPE_CHECKING:
    from ...agent import Agent

logger = logging.getLogger(__name__)

OTHER_OPTION = "other"
"""The option the decision model picks when no configured action fits; the LLM then runs."""

TOOL_USE_ID_PREFIX = "s1-"
"""Prefix of every ``toolUseId`` the fast path synthesizes, so an audit can find them without the metadata."""

_DEFAULT_INSTRUCTIONS = (
    "Which single action in the options should the agent take next to progress `request`, given "
    "`agent_instructions` and `latest_tool_result`? Choose `other` when none fits, when the request is complete, or "
    "when the next step needs reasoning a fixed action cannot express."
)
_MEDIA_LABELS = {"image": "[Image]", "document": "[Document]", "video": "[Video]"}


@dataclass(frozen=True)
class ToolCall:
    """A fixed tool call the fast path may emit for one option.

    Args:
        name: Name of a tool registered on the agent.
        input: The tool input, sent verbatim (a copy per call).
        description: What choosing this action means, shown to the decision model. Defaults to the call itself.
    """

    name: str
    input: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None

    def _option_description(self) -> str:
        if self.description:
            return self.description
        return json.dumps({"tool": self.name, "input": dict(self.input)}, ensure_ascii=False, default=str)


class FastPath(Plugin):
    """Serve a model call from a System One decision model when the next step is one of a few known actions.

    Before each model call, one ``Choice`` over ``actions`` plus ``other`` is asked about the conversation. A
    confident answer (at or above ``min_confidence``) short-circuits the call: the agent receives a synthesized
    assistant turn holding that action's tool call, with stop reason ``tool_use``, and runs the tool normally, so tool
    hooks, interventions and tool spans all apply. ``other``, a below-floor answer, or a decision error runs the LLM
    unchanged.

    A synthesized turn is never passed off as model output. ``message["metadata"]["custom"]["strands"]`` records
    ``{"source": "decision", "decision_model": ..., "confidence": ...}``, the ``toolUseId`` starts with ``s1-``, the
    decision span (with ``strands.fast_path.action``) stands in for the cycle's ``chat`` span, no
    ``AfterModelCallEvent`` fires, and no LLM usage is billed for the cycle.
    """

    name = "strands:fast-path"

    def __init__(
        self,
        decision_model: DecisionModel,
        actions: Mapping[str, ToolCall],
        *,
        min_confidence: float,
        instructions: str = _DEFAULT_INSTRUCTIONS,
        max_consecutive: int = 3,
        max_request_tokens: int = 1_000,
        max_instruction_tokens: int = 1_000,
    ) -> None:
        """Initialize the fast path.

        Args:
            decision_model: A calibrated decision model.
            actions: Option name to the tool call it emits. ``other`` is reserved.
            min_confidence: Serve an action only at or above this confidence. Required: an unconditional fast path
                would drive the agent open-loop.
            instructions: The question asked before each model call.
            max_consecutive: Force an LLM turn after this many consecutive synthesized turns.
            max_request_tokens: Budget for the request text and for the latest tool result sent as state.
            max_instruction_tokens: Budget for the agent's system-prompt text sent as state.

        Raises:
            ValueError: If the model is uncalibrated, ``actions`` is empty or uses ``other``, ``min_confidence`` is
                outside (0, 1], or a bound is not a positive integer.
            TypeError: If an action is not a ``ToolCall``.
        """
        _require_calibrated("FastPath", decision_model, min_confidence)
        _validate_actions(actions)
        if isinstance(min_confidence, bool) or not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be greater than 0 and at most 1")
        for name, value in (
            ("max_consecutive", max_consecutive),
            ("max_request_tokens", max_request_tokens),
            ("max_instruction_tokens", max_instruction_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._model = decision_model
        self._actions = dict(actions)
        self._min_confidence = min_confidence
        self._max_consecutive = max_consecutive
        self._request_tokens = max_request_tokens
        self._instruction_tokens = max_instruction_tokens
        self._question = Choice(
            instructions,
            options={
                **{option: action._option_description() for option, action in self._actions.items()},
                OTHER_OPTION: "None of the actions fits; let the reasoning model decide",
            },
        )
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        """Register the fast-path middleware on the agent's model invocation.

        An action whose tool is not offered on a given call is never served, so tools a provider loads later work.
        """
        agent._middleware_registry.add_middleware(InvokeModelStage, self._middleware)

    async def _middleware(self, context: InvokeModelContext, next_fn: Any) -> AsyncGenerator[Any, None]:
        message = await self._serve(context)
        if message is None:
            async for event in next_fn(context):
                yield event
            return
        yield message

    async def _serve(self, context: InvokeModelContext) -> ModelStopReason | None:
        """Return the synthesized turn's stop event, or None to run the model."""
        if not self._eligible(context):
            return None
        chosen: dict[str, str] = {}
        started = time.perf_counter()
        try:
            response = await self._model._traced_ask(
                self._state(context),
                {"action": self._question},
                _span_attributes=lambda response: self._span_attributes(response, context, chosen),
            )
        except Exception as error:
            logger.warning(
                "adapter=<FastPath>, reason=<decision_error>, error_type=<%s> | passing through to the model",
                type(error).__name__,
            )
            return None
        if "action" not in chosen:
            return None
        answer = response.answers["action"]
        assert isinstance(answer, ChoiceAnswer)
        return self._synthesize(chosen["action"], answer, response, time.perf_counter() - started)

    def _eligible(self, context: InvokeModelContext) -> bool:
        # A forced tool choice (structured output) must reach the model.
        if context.tool_choice is not None:
            return False
        if _consecutive_synthesized(context.messages) >= self._max_consecutive:
            logger.debug("max_consecutive=<%d> | forcing a model turn", self._max_consecutive)
            return False
        return True

    def _span_attributes(
        self, response: DecisionResponse, context: InvokeModelContext, chosen: dict[str, str]
    ) -> dict[str, Any]:
        """Decide whether to serve, recording the outcome on the decision span; fills ``chosen`` when served."""
        answer = response.answers["action"]
        assert isinstance(answer, ChoiceAnswer)  # ask() checks answer types before this runs
        reason = self._decline_reason(answer, context)
        if reason is not None:
            logger.debug(
                "choice=<%s>, confidence=<%s>, reason=<%s> | passing through to the model",
                answer.choice,
                answer.confidence,
                reason,
            )
            return {"strands.fast_path.served": False, "strands.fast_path.reason": reason}
        chosen["action"] = answer.choice
        return {"strands.fast_path.served": True, "strands.fast_path.action": answer.choice}

    def _decline_reason(self, answer: ChoiceAnswer, context: InvokeModelContext) -> str | None:
        if answer.choice == OTHER_OPTION:
            return "other"
        if answer.confidence is None or answer.confidence < self._min_confidence:
            return "below_floor"
        offered = {spec["name"] for spec in context.tool_specs}
        if self._actions[answer.choice].name not in offered:
            return "tool_not_offered"
        return None

    def _state(self, context: InvokeModelContext) -> dict[str, str]:
        state = project_state(
            context.messages,
            context.system_prompt,
            max_tokens=self._request_tokens,
            max_instruction_tokens=self._instruction_tokens,
        )
        result = _latest_tool_result_text(context.messages, self._request_tokens * CHARS_PER_TOKEN)
        if result is not None:
            state["latest_tool_result"] = result
        return state

    def _synthesize(
        self, option: str, answer: ChoiceAnswer, response: DecisionResponse, elapsed: float
    ) -> ModelStopReason:
        action = self._actions[option]
        message: Message = {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": f"{TOOL_USE_ID_PREFIX}{uuid.uuid4()}",
                        "name": action.name,
                        "input": copy.deepcopy(dict(action.input)),
                    }
                }
            ],
        }
        mark_synthesized(
            message,
            DECISION_SOURCE,
            decision_model=response.model_id or self._model.model_id,
            confidence=answer.confidence,
        )
        logger.debug("action=<%s>, confidence=<%s> | fast path served the model call", option, answer.confidence)
        return ModelStopReason(
            stop_reason="tool_use",
            message=message,
            usage={"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            metrics={"latencyMs": round(elapsed * 1000)},
        )


def _validate_actions(actions: Mapping[str, ToolCall]) -> None:
    if not actions:
        raise ValueError("FastPath needs at least one action")
    if OTHER_OPTION in actions:
        raise ValueError(f"'{OTHER_OPTION}' is reserved for the pass-through option; rename that action")
    for option, action in actions.items():
        if not isinstance(action, ToolCall):
            raise TypeError(f"action {option!r} must be a ToolCall, got {type(action).__name__}")


def _consecutive_synthesized(messages: Messages) -> int:
    """Count the trailing assistant turns the fast path synthesized, up to the last model-generated one."""
    count = 0
    for message in reversed(messages):
        if message["role"] != "assistant":
            continue
        if synthesized_source(message) != DECISION_SOURCE:
            break
        count += 1
    return count


def _latest_tool_result_text(messages: Messages, character_limit: int) -> str | None:
    """Bounded text of the tool results in the latest message, when it carries them."""
    if not messages or messages[-1]["role"] != "user":
        return None
    parts: list[str] = []
    for block in messages[-1]["content"]:
        result = block.get("toolResult")
        if result is None:
            continue
        if result.get("status") == "error":
            parts.append("[error]")
        for item in result.get("content", []):
            parts.extend(_result_item_text(item))
    if not parts:
        return None
    return truncate_text("\n".join(parts), character_limit)


def _result_item_text(item: Mapping[str, Any]) -> list[str]:
    parts = []
    if isinstance(item.get("text"), str):
        parts.append(item["text"])
    if "json" in item:
        parts.append(json.dumps(item["json"], ensure_ascii=False, default=str))
    parts.extend(label for kind, label in _MEDIA_LABELS.items() if kind in item)
    return parts
