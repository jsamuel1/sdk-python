"""Swarm handoff with a System One decision model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from ...agent.agent_result import AgentResult
from ...models._request_text import CHARS_PER_TOKEN, project_state, truncate_text
from ._agent import DECISION_STATE_KEY
from ._model import DecisionModel
from ._strategy import _require_calibrated, candidate_evidence
from ._types import Choice, ChoiceAnswer, Decision

if TYPE_CHECKING:
    from ...multiagent.swarm import Handoff, HandoffContext, SwarmNode

logger = logging.getLogger(__name__)

COMPLETE_OPTION = "complete"
"""The option that ends the swarm; a ``HandoffDecision.next`` of this value means no handoff."""

HANDOFF_FIELD = "next"
"""Answer key of the handoff decision recorded on the node's result."""

_DEFAULT_INSTRUCTIONS = (
    "`request` is the task given to the agent that just ran, `agent_instructions` describe that agent, and "
    "`response` is its final answer. Which candidate agent should act next? Choose `complete` when `response` "
    "fully resolves `request` and no other agent is needed. Treat missing candidate evidence as unknown."
)
_COMPLETE_EVIDENCE = "The task is finished: `response` resolves `request`, so no other agent should act."
_STAY_MESSAGE = (
    "A decision model could not choose the next agent with enough confidence, so you continue. Finish the task, "
    "or call handoff_to_agent if another agent should take over."
)


class HandoffDecision(BaseModel):
    """The output of a handoff decision: the next node's id, or ``"complete"``."""

    next: str


class DecisionHandoffStrategy:
    """Choose a swarm's next node with one System One ``Choice`` over the other nodes plus ``complete``.

    Runs only when a node finished without calling ``handoff_to_agent``; an explicit handoff always wins. The
    candidates' evidence is their name and description, projected exactly as ``DecisionStrategy`` projects router
    candidates. The decision is recorded on the node's result under ``state["decision"]``.

    Below ``min_confidence``, or when the decision request fails, ``fallback`` applies: ``"complete"`` ends the swarm
    (the behaviour without a strategy) and ``"stay"`` re-runs the node with a note that routing was unsure. A
    re-run counts against ``max_handoffs`` and ``max_iterations`` like any handoff.

    Experimental: subject to change without notice.
    """

    def __init__(
        self,
        decision_model: DecisionModel,
        *,
        min_confidence: float | None = None,
        fallback: Literal["complete", "stay"] = "complete",
        instructions: str = _DEFAULT_INSTRUCTIONS,
        max_request_tokens: int = 1_000,
        max_instruction_tokens: int = 1_000,
        max_response_tokens: int = 1_000,
    ) -> None:
        """Initialize the strategy.

        Args:
            decision_model: The model that makes the handoff decision.
            min_confidence: Apply ``fallback`` when the choice's confidence is below this. Requires a calibrated
                model.
            fallback: What to do when the decision is below the floor or fails: ``"complete"`` or ``"stay"``.
            instructions: The handoff question. Candidate evidence, the request and the response are sent as state.
            max_request_tokens: Budget for the node's latest request text sent as state.
            max_instruction_tokens: Budget for the node's system-prompt text sent as state.
            max_response_tokens: Budget for the node's final response text, sent as state and in the handoff
                message.

        Raises:
            ValueError: If ``min_confidence`` is set on an uncalibrated model, ``fallback`` is unknown, or a budget
                is not positive.
        """
        _require_calibrated("DecisionHandoffStrategy", decision_model, min_confidence)
        if fallback not in ("complete", "stay"):
            raise ValueError(f"fallback must be 'complete' or 'stay', got {fallback!r}")
        if min(max_request_tokens, max_instruction_tokens, max_response_tokens) <= 0:
            raise ValueError("token budgets must be greater than zero")
        self._model = decision_model
        self._min_confidence = min_confidence
        self._fallback = fallback
        self._instructions = instructions
        self._request_tokens = max_request_tokens
        self._instruction_tokens = max_instruction_tokens
        self._response_characters = max_response_tokens * CHARS_PER_TOKEN

    async def select(self, context: HandoffContext) -> Handoff | None:
        """Return the handoff the decision model chose, the fallback, or None to complete the swarm.

        Raises:
            ValueError: If a candidate node is named ``"complete"``.
        """
        if not context.candidates:
            return None
        keys = _option_keys(context.candidates)
        response_text = _response_text(context.result, self._response_characters)
        question = Choice(self._instructions, options=_options(keys))
        executor = context.current.executor
        state: dict[str, Any] = {
            **project_state(
                executor.messages,
                executor.system_prompt,
                max_tokens=self._request_tokens,
                max_instruction_tokens=self._instruction_tokens,
            ),
            "response": response_text,
        }
        try:
            response = await self._model.ask(state, {HANDOFF_FIELD: question})
        except Exception as error:
            logger.warning(
                "strategy=<%s>, node=<%s>, fallback=<%s>, error_type=<%s> | handoff decision failed, using fallback",
                type(self).__name__,
                context.current.node_id,
                self._fallback,
                type(error).__name__,
            )
            return self._fallback_handoff(context, response_text)
        answer = _by_node_id(response.answers[HANDOFF_FIELD], keys)
        _record(context, answer, response.model_id, response.usage)
        if self._min_confidence is not None and (answer.confidence or 0.0) < self._min_confidence:
            logger.debug(
                "choice=<%s>, confidence=<%s>, min_confidence=<%s>, fallback=<%s> | handoff decision below floor",
                answer.choice,
                answer.confidence,
                self._min_confidence,
                self._fallback,
            )
            return self._fallback_handoff(context, response_text)
        if answer.choice == COMPLETE_OPTION:
            return None
        target = next(node for node in context.candidates if node.node_id == answer.choice)
        confidence = "" if answer.confidence is None else f" (confidence {answer.confidence:.2f})"
        routed = f"{context.current.node_id} finished without handing off; a decision model routed the task to you"
        return _handoff(target, f"{routed}{confidence}.", context.current, response_text)

    def _fallback_handoff(self, context: HandoffContext, response_text: str) -> Handoff | None:
        if self._fallback == "complete":
            return None
        return _handoff(context.current, _STAY_MESSAGE, context.current, response_text)


def _option_keys(candidates: Any) -> dict[str, SwarmNode | None]:
    keys: dict[str, SwarmNode | None] = {}
    for index, node in enumerate(candidates):
        if node.node_id == COMPLETE_OPTION:
            raise ValueError(f"swarm node id {COMPLETE_OPTION!r} collides with the handoff decision's complete option")
        keys[f"c{index}"] = node
    keys[COMPLETE_OPTION] = None
    return keys


def _options(keys: Mapping[str, SwarmNode | None]) -> dict[str, str | None]:
    return {
        key: _COMPLETE_EVIDENCE if node is None else candidate_evidence(node.node_id, node.executor.description)
        for key, node in keys.items()
    }


def _by_node_id(answer: Any, keys: Mapping[str, SwarmNode | None]) -> ChoiceAnswer:
    """Re-key a Choice answer from the option keys sent to the model to node ids (and ``complete``)."""
    assert isinstance(answer, ChoiceAnswer)

    def name(key: str) -> str:
        node = keys[key]
        return COMPLETE_OPTION if node is None else node.node_id

    return ChoiceAnswer(
        choice=name(answer.choice),
        probabilities={name(key): value for key, value in answer.probabilities.items()},
        confidence=answer.confidence,
        logits=None if answer.logits is None else {name(key): value for key, value in answer.logits.items()},
        extras=answer.extras,
    )


def _record(context: HandoffContext, answer: ChoiceAnswer, model_id: str | None, usage: Any) -> None:
    agent_result = context.result.result
    if not isinstance(agent_result, AgentResult) or not isinstance(agent_result.state, dict):
        return
    result_state = agent_result.state
    decision = Decision(
        output=HandoffDecision(next=answer.choice), answers={HANDOFF_FIELD: answer}, model_id=model_id, usage=usage
    )
    # An Agent's result state is the invocation's request_state, which every swarm node shares; a copy keeps
    # the decision on this node's result only.
    agent_result.state = {**result_state, DECISION_STATE_KEY: decision}


def _handoff(target: SwarmNode, message: str, previous: SwarmNode, response_text: str) -> Handoff:
    """Build a handoff whose message carries the previous node's response, which the next node cannot otherwise see."""
    from ...multiagent.swarm import Handoff

    if response_text:
        message = f"{message}\n\n{previous.node_id}'s final response:\n{response_text}"
    return Handoff(node=target, message=message)


def _response_text(node_result: Any, character_limit: int) -> str:
    message = getattr(node_result.result, "message", None)
    content = message.get("content", []) if isinstance(message, Mapping) else []
    text = "\n".join(block["text"] for block in content if isinstance(block.get("text"), str))
    return truncate_text(text, character_limit)
