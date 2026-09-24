"""System One decisions from a SageMaker real-time endpoint that serves TypeSafe's ``/v1/systemone`` wire.

For self-hosted System One models (for example Kev or CLM) deployed in your own AWS account. The endpoint's
container maps ``/invocations`` to ``/v1/systemone``, so the body is exactly the wire ``TypeSafeDecisionModel``
sends; SigV4 ``InvokeEndpoint`` carries it in place of a bearer key. Request and answer mapping are shared with
``TypeSafeDecisionModel``.

- Docs: https://docs.typesafe.ai/api
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Mapping
from typing import Any, TypedDict

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from pydantic import TypeAdapter, ValidationError
from typing_extensions import Unpack, override

from ..experimental.decisions import DecisionModel, DecisionResponse, DecisionState, Question
from ..experimental.decisions._types import Answer
from ..types.event_loop import Usage
from ..types.exceptions import ModelThrottledException
from ._validation import validate_config_keys
from .typesafe import (
    DEFAULT_MODEL_ID,
    MAX_REQUEST_TOKENS,
    MAX_STATE_PLUS_QUESTION_TOKENS,
    _check_budget,
    _check_calibration_config,
    _check_limits,
    _from_vendor,
    _to_wire,
    typesafe_sdk,
)

logger = logging.getLogger(__name__)

# SageMaker Runtime error codes that mean "try again later", not "this request is wrong".
_THROTTLE_CODES = frozenset({"ThrottlingException", "ServiceUnavailable", "ModelNotReadyException"})
_SERVER_LATENCY_KEY = "server_latency_ms"


class SageMakerDecisionModel(DecisionModel):
    """System One decisions from a SageMaker endpoint that speaks TypeSafe's ``/v1/systemone`` API.

    Calibration is part of the configured instance, as for ``TypeSafeDecisionModel``: ``temperature=1.0`` (the
    identity) and ``calibrated=True`` pass the server's probabilities through; a fitted ``temperature=`` rescales
    them on the client, and ``calibrated=False`` makes confidence gates refuse the instance. A server-reported
    ``server_latency_ms`` in the response body is kept on every answer's ``extras``.
    """

    class SageMakerDecisionConfig(TypedDict, total=False):
        """Configuration for SageMaker decision models.

        Attributes:
            endpoint_name: The SageMaker real-time endpoint to invoke (set by the constructor; updatable).
            model_id: Model name sent in the request body's ``model`` field.
            max_state_plus_question_tokens: Estimated-token budget for the state plus the longest question.
            max_request_tokens: Estimated-token budget for the whole request, or ``None`` for no cap.
            temperature: Applied on the client on top of the server's probabilities; 1.0 is the identity.
            calibrated: Whether this instance's confidences are calibrated.
        """

        endpoint_name: str
        model_id: str
        max_state_plus_question_tokens: int
        max_request_tokens: int | None
        temperature: float
        calibrated: bool

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str | None = None,
        boto_session: boto3.Session | None = None,
        boto_client_config: BotocoreConfig | None = None,
        client: Any | None = None,
        **model_config: Unpack[_ModelConfig],
    ) -> None:
        """Initialize the provider.

        Args:
            endpoint_name: The SageMaker real-time endpoint to invoke.
            region: AWS region of the endpoint. Defaults to the session's region.
            boto_session: Boto session for credentials and region. Defaults to a new ``boto3.Session``.
            boto_client_config: Extra botocore configuration for the SageMaker Runtime client.
            client: A preconfigured ``sagemaker-runtime`` client; when given, ``region``, ``boto_session`` and
                ``boto_client_config`` are ignored.
            **model_config: Model configuration; see ``SageMakerDecisionConfig`` (every key but ``endpoint_name``).

        Raises:
            ValueError: If both ``region`` and ``boto_session`` are given, ``temperature`` is not a finite number
                above 0, or ``calibrated`` is not a bool.
        """
        validate_config_keys(model_config, _ModelConfig)
        _check_calibration_config(model_config)
        if region and boto_session:
            raise ValueError("Cannot specify both `region` and `boto_session`.")
        self.config = SageMakerDecisionModel.SageMakerDecisionConfig(
            endpoint_name=endpoint_name,
            model_id=DEFAULT_MODEL_ID,
            max_state_plus_question_tokens=MAX_STATE_PLUS_QUESTION_TOKENS,
            max_request_tokens=MAX_REQUEST_TOKENS,
            temperature=1.0,
            calibrated=True,
        )
        self.config.update(model_config)
        self._client = client if client is not None else _runtime_client(region, boto_session, boto_client_config)

    @property
    @override
    def calibrated(self) -> bool:
        """The configured ``calibrated`` value."""
        return self.config["calibrated"]

    @override
    def update_config(self, **model_config: Unpack[SageMakerDecisionConfig]) -> None:  # type: ignore[override]
        """Update the model configuration.

        Raises:
            ValueError: If ``temperature`` is not a finite number above 0, or ``calibrated`` is not a bool.
        """
        validate_config_keys(model_config, self.SageMakerDecisionConfig)
        _check_calibration_config(model_config)
        self.config.update(model_config)

    @override
    def get_config(self) -> SageMakerDecisionConfig:
        """Return the model configuration."""
        return self.config

    @override
    async def _ask(self, state: DecisionState, questions: Mapping[str, Question], **kwargs: Any) -> DecisionResponse:
        """Answer every question in one ``InvokeEndpoint`` call.

        Raises:
            ContextWindowOverflowException: If the request exceeds the configured context budgets; raised before
                sending.
            ModelThrottledException: On SageMaker throttling, a busy service, or a model that is not ready.
            ValueError: If a question exceeds the wire's option or level limits, or the endpoint returns a body
                that is not a ``/v1/systemone`` response.
        """
        _check_limits(questions)
        _check_budget(
            state,
            questions,
            max_row=self.config["max_state_plus_question_tokens"],
            max_request=self.config.get("max_request_tokens"),
        )
        body = {
            "state": state,
            "model": self.config["model_id"],
            "questions": {question_id: _to_wire(question) for question_id, question in questions.items()},
        }
        payload = await self._invoke(json.dumps(body, ensure_ascii=False, default=str))
        return self._response(payload)

    async def _invoke(self, body: str) -> dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                self._client.invoke_endpoint,
                EndpointName=self.config["endpoint_name"],
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in _THROTTLE_CODES:
                raise ModelThrottledException(str(error)) from error
            raise
        payload = json.loads(response["Body"].read())
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
            raise ValueError("SageMaker decision endpoint returned no answers")
        return payload

    def _response(self, payload: Mapping[str, Any]) -> DecisionResponse:
        temperature, calibrated = self.config["temperature"], self.config["calibrated"]
        latency = payload.get(_SERVER_LATENCY_KEY)
        answers = {
            question_id: _with_latency(_from_vendor(_parse_answer(question_id, raw), temperature, calibrated), latency)
            for question_id, raw in payload["answers"].items()
        }
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        model_id = str(payload.get("model") or self.config["model_id"])
        logger.debug("model=<%s>, questions=<%d> | sagemaker decision answered", model_id, len(answers))
        return DecisionResponse(
            answers=answers,
            model_id=model_id,
            usage=Usage(inputTokens=input_tokens, outputTokens=output_tokens, totalTokens=input_tokens + output_tokens),
        )


class _ModelConfig(TypedDict, total=False):
    """``SageMakerDecisionConfig`` minus ``endpoint_name``, which the constructor takes explicitly."""

    model_id: str
    max_state_plus_question_tokens: int
    max_request_tokens: int | None
    temperature: float
    calibrated: bool


_ANSWER_ADAPTER: TypeAdapter[Any] = TypeAdapter(typesafe_sdk.Answer)


def _parse_answer(question_id: str, raw: Any) -> Any:
    """Validate one wire answer with the TypeSafe SDK's own models, so both providers share one decoder."""
    try:
        # JSON mode, as the SDK decodes its own responses: its strict models accept "0"-style score keys only there.
        return _ANSWER_ADAPTER.validate_json(json.dumps(raw))
    except ValidationError as error:
        raise ValueError(
            f"{question_id}: not a /v1/systemone answer ({error.error_count()} validation errors)"
        ) from error


def _with_latency(answer: Answer, latency: Any) -> Answer:
    if latency is None:
        return answer
    return dataclasses.replace(answer, extras={**answer.extras, _SERVER_LATENCY_KEY: latency})


def _runtime_client(region: str | None, session: boto3.Session | None, config: BotocoreConfig | None) -> Any:
    session = session or boto3.Session(region_name=region)
    existing = getattr(config, "user_agent_extra", None)
    agent = f"{existing} strands-agents" if existing else "strands-agents"
    client_config = (config or BotocoreConfig()).merge(BotocoreConfig(user_agent_extra=agent))
    return session.client("sagemaker-runtime", config=client_config)
