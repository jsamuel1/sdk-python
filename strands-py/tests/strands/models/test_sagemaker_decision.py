import io
import json
from typing import Any

import pytest
from botocore.exceptions import ClientError

from strands.experimental.decisions import (
    Choice,
    ChoiceAnswer,
    DecisionStrategy,
    Score,
    ScoreAnswer,
    YesNo,
    YesNoAnswer,
)
from strands.models.sagemaker_decision import SageMakerDecisionModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

QUESTIONS = {
    "dept": Choice("Which team?", options={"billing": "Payments", "technical": None}),
    "urgent": YesNo("Urgent?", true="Time pressure", false="Can wait"),
    "anger": Score("How angry?", levels=["calm", "upset", "furious"]),
}
PAYLOAD = {
    "model": "clm-v0.1-8b",
    "answers": {
        "dept": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.9, "technical": 0.1},
        },
        "urgent": {"type": "noul", "noul": 0.95},
        "anger": {
            "type": "score",
            "score": 1.1,
            "confidence": 0.5,
            "legend": {"0": "calm", "1": "upset", "2": "furious"},
            "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
        },
    },
    "usage": {"input_tokens": 42, "output_tokens": 3},
}


class _Runtime:
    """A stubbed ``sagemaker-runtime`` client: records every call and returns or raises what it is given."""

    def __init__(self, payload: Any = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def invoke_endpoint(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"Body": io.BytesIO(json.dumps(self.payload).encode())}


def _model(runtime: _Runtime, **config: Any) -> SageMakerDecisionModel:
    return SageMakerDecisionModel(endpoint_name="clm-v01-8b", client=runtime, model_id="clm-latest", **config)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "busy"}}, "InvokeEndpoint")


@pytest.mark.asyncio
async def test_request_body_is_the_systemone_wire():
    runtime = _Runtime(PAYLOAD)

    await _model(runtime).ask({"ticket": "charged twice"}, QUESTIONS)

    call = runtime.calls[0]
    assert call["EndpointName"] == "clm-v01-8b"
    assert call["ContentType"] == "application/json" and call["Accept"] == "application/json"
    assert json.loads(call["Body"]) == {
        "state": {"ticket": "charged twice"},
        "model": "clm-latest",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments", "technical": None},
            },
            "urgent": {
                "type": "noul",
                "instructions": "Urgent?",
                "criteria": {"true": "Time pressure", "false": "Can wait"},
            },
            "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "upset", "furious"]},
        },
    }


@pytest.mark.asyncio
async def test_temperature_one_passes_the_answers_through():
    response = await _model(_Runtime(PAYLOAD)).ask("s", QUESTIONS)

    assert response.answers == {
        "dept": ChoiceAnswer(choice="billing", probabilities={"billing": 0.9, "technical": 0.1}, confidence=0.8),
        "urgent": YesNoAnswer(probability=0.95, confidence=pytest.approx(0.9)),
        "anger": ScoreAnswer(score=1.1, probabilities={0: 0.1, 1: 0.7, 2: 0.2}, confidence=0.5),
    }
    assert response.model_id == "clm-v0.1-8b"
    assert response.usage == {"inputTokens": 42, "outputTokens": 3, "totalTokens": 45}


@pytest.mark.asyncio
async def test_temperature_rescales_and_keeps_the_server_numbers():
    response = await _model(_Runtime(PAYLOAD), temperature=2.0).ask("s", QUESTIONS)

    sq = {"billing": 0.9**0.5, "technical": 0.1**0.5}
    exp_billing = sq["billing"] / sum(sq.values())
    assert response["dept"].probabilities["billing"] == pytest.approx(exp_billing)
    assert response["dept"].confidence == pytest.approx(exp_billing)
    assert response["dept"].extras == {"confidence": 0.8, "probabilities": {"billing": 0.9, "technical": 0.1}}
    assert response["urgent"].extras == {"noul": 0.95}


@pytest.mark.asyncio
async def test_uncalibrated_instance_reports_no_confidence_and_gates_refuse_it():
    model = _model(_Runtime(PAYLOAD), calibrated=False)

    response = await model.ask("s", QUESTIONS)

    assert model.calibrated is False
    assert response["dept"].confidence is None and response["anger"].confidence is None
    with pytest.raises(ValueError, match=r"configure a raw-logit provider with temperature="):
        DecisionStrategy(model, min_confidence=0.7)


@pytest.mark.asyncio
async def test_server_latency_goes_on_every_answer_not_on_the_model():
    model = _model(_Runtime({**PAYLOAD, "server_latency_ms": 12.5}))

    response = await model.ask("s", QUESTIONS)

    assert {answer.extras["server_latency_ms"] for answer in response.answers.values()} == {12.5}
    assert not any(name.startswith("last_") for name in vars(model))


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["ThrottlingException", "ServiceUnavailable", "ModelNotReadyException"])
async def test_throttle_codes_map_to_model_throttled(code):
    with pytest.raises(ModelThrottledException):
        await _model(_Runtime(error=_client_error(code))).ask("s", QUESTIONS)


@pytest.mark.asyncio
async def test_other_client_errors_propagate():
    with pytest.raises(ClientError, match="ValidationError"):
        await _model(_Runtime(error=_client_error("ValidationError"))).ask("s", QUESTIONS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"error": "bad"}, "returned no answers"),
        (["not", "an", "object"], "returned no answers"),
        ({"answers": {"dept": {"type": "extract"}}}, "dept: not a /v1/systemone answer"),
    ],
)
async def test_malformed_bodies_are_refused(payload, message):
    with pytest.raises(ValueError, match=message):
        await _model(_Runtime(payload)).ask("s", {"dept": QUESTIONS["dept"]})


@pytest.mark.asyncio
async def test_budget_is_checked_before_invoking():
    runtime = _Runtime(PAYLOAD)

    with pytest.raises(ContextWindowOverflowException):
        await _model(runtime, max_state_plus_question_tokens=10).ask("x" * 400, QUESTIONS)
    assert runtime.calls == []


def test_config_round_trips_and_validates():
    model = _model(_Runtime(), temperature=1.4)

    model.update_config(temperature=2.0, calibrated=False)

    assert model.get_config() == {
        "endpoint_name": "clm-v01-8b",
        "model_id": "clm-latest",
        "max_state_plus_question_tokens": 32_000,
        "max_request_tokens": 64_000,
        "temperature": 2.0,
        "calibrated": False,
    }
    with pytest.raises(ValueError, match="temperature must be"):
        model.update_config(temperature=-1.0)
    with pytest.raises(ValueError, match="temperature must be"):
        SageMakerDecisionModel(endpoint_name="e", client=_Runtime(), temperature=0)


def test_unknown_config_keys_warn():
    with pytest.warns(UserWarning, match="Invalid configuration parameters"):
        SageMakerDecisionModel(endpoint_name="e", client=_Runtime(), top_p=0.1)  # type: ignore[call-arg]


def test_region_and_session_are_exclusive_and_default_client_is_built():
    import boto3

    with pytest.raises(ValueError, match="Cannot specify both"):
        SageMakerDecisionModel(endpoint_name="e", region="us-west-2", boto_session=boto3.Session())

    model = SageMakerDecisionModel(endpoint_name="e", region="us-west-2")

    assert model._client.meta.region_name == "us-west-2"
    assert "strands-agents" in model._client.meta.config.user_agent_extra


def test_models_package_exports_it_lazily():
    import strands.models as models

    assert models.SageMakerDecisionModel is SageMakerDecisionModel
