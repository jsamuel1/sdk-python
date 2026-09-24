"""Expose a decision schema as a tool an LLM can call for a calibrated judgment mid-task."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from pydantic import BaseModel, ValidationError
from typing_extensions import override

from ...types._events import ToolResultEvent
from ...types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse
from ._model import DecisionModel

logger = logging.getLogger(__name__)


class DecisionTool(AgentTool):
    """A tool that asks a decision model a fixed schema about the state the calling LLM supplies.

    The result carries every answer's probabilities and confidence back to the LLM, so the model sees the
    uncertainty and decides what to do with it. Of all placements this is the only one the LLM may skip; put
    decisions that must always happen in a guard or a ``DecisionAgent`` instead.
    """

    def __init__(
        self,
        decision_model: DecisionModel,
        schema: type[BaseModel],
        *,
        state_schema: type[BaseModel],
        name: str,
        description: str,
    ) -> None:
        """Initialize the tool.

        Args:
            decision_model: The decision model.
            schema: The decision schema asked on every call.
            state_schema: Pydantic model of the tool input; the validated input becomes the decision state.
            name: Tool name shown to the LLM.
            description: Tool description shown to the LLM; say what the judgment is for.
        """
        super().__init__()
        self._model = decision_model
        self._schema = schema
        self._state_schema = state_schema
        self._name = name
        self._description = description

    @property
    def tool_name(self) -> str:
        """The tool name."""
        return self._name

    @property
    def tool_spec(self) -> ToolSpec:
        """Tool specification whose input schema is ``state_schema``."""
        return {
            "name": self._name,
            "description": self._description,
            "inputSchema": {"json": self._state_schema.model_json_schema()},
        }

    @property
    def tool_type(self) -> str:
        """The tool type."""
        return "decision"

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Validate the input, ask the schema, and yield the answers as a JSON tool result."""
        tool_use_id = tool_use["toolUseId"]
        try:
            state = self._state_schema.model_validate(tool_use["input"]).model_dump(mode="json")
        except ValidationError as error:
            # The LLM wrote this input, so it gets the full validation message to fix it.
            yield _error(tool_use_id, f"{self._name} could not decide: {error}")
            return
        try:
            decision = await self._model.decide(self._schema, state)
        except Exception as error:
            # Decision errors, including a provider's ValueError for a rejected request, can carry URLs, request ids
            # or other internals: log them, and give the LLM only the error type.
            logger.warning(
                "tool_name=<%s>, error_type=<%s>, error=<%s> | decision failed", self._name, type(error).__name__, error
            )
            yield _error(
                tool_use_id, f"{self._name} could not decide: the decision service failed ({type(error).__name__})"
            )
            return
        payload = {
            "output": decision.output.model_dump(mode="json"),
            "answers": {field: _answer_json(answer) for field, answer in decision.answers.items()},
            "model_id": decision.model_id,
        }
        yield ToolResultEvent({"toolUseId": tool_use_id, "status": "success", "content": [{"json": payload}]})


def _answer_json(answer: Any) -> dict[str, Any]:
    """An answer as JSON: derived fields always, ``logits``/``extras`` only when the provider supplied them."""
    data = {key: value for key, value in dataclasses.asdict(answer).items() if key not in ("logits", "extras")}
    if answer.logits is not None:
        data["logits"] = {_json_key(key): value for key, value in answer.logits.items()}
    if answer.extras:
        data["extras"] = dict(answer.extras)
    return data


def _json_key(key: Any) -> Any:
    # YesNo logits are keyed True/False; JSON object keys must be strings.
    return str(key).lower() if isinstance(key, bool) else key


def _error(tool_use_id: str, message: str) -> ToolResultEvent:
    return ToolResultEvent({"toolUseId": tool_use_id, "status": "error", "content": [{"text": message}]})


def decision_tool(
    decision_model: DecisionModel,
    schema: type[BaseModel],
    *,
    state_schema: type[BaseModel],
    name: str,
    description: str,
) -> DecisionTool:
    """Build a tool that asks ``schema`` about state the calling LLM supplies (see ``DecisionTool``)."""
    return DecisionTool(decision_model, schema, state_schema=state_schema, name=name, description=description)
