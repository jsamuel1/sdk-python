"""LLMDecisionModel answers from token logprobs (design 0020 D23(k)).

The fixtures are real Bedrock Converse responses (forced tool, ``top_logprobs=20``) recorded for the logprobs research
(``.agents/scratchpad/logprobs/transcripts/strands-shape-20260928T065107Z.jsonl``); only ``-9999`` entries are dropped.
"""

import json
import math
import pathlib
import unittest.mock
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

import strands.models.bedrock
from strands.experimental.decisions import (
    Choice,
    ChoiceAnswer,
    DecisionStrategy,
    LLMDecisionModel,
    Score,
    ScoreAnswer,
    YesNo,
    YesNoAnswer,
    fit_temperature,
)
from strands.experimental.decisions._logprobs import label_logits, logprob_tokens
from strands.models.bedrock import BedrockModel
from tests.fixtures.mocked_model_provider import MockedModelProvider

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _fixture(name):
    return json.loads((_FIXTURES / f"converse_logprobs_{name}.json").read_text())


class Ticket(BaseModel):
    dept: Annotated[Literal["billing", "shipping", "technical"], Choice("Which department?")]
    anger: Annotated[float, Score("How angry?", levels=["calm", "annoyed", "frustrated", "angry", "furious"])]
    cancel: Annotated[bool, YesNo("Threatens to cancel?")]


class _LogprobModel(MockedModelProvider):
    """Yields a Converse ``messageStop`` stream event carrying ``content`` logprobs, then the structured output."""

    def __init__(self, values, content, *, fields=None):
        super().__init__([])
        self.values = values
        self.fields = {"choices": [{"logprobs": {"content": content}}]} if fields is None else fields

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"event": {"messageStop": {"stopReason": "tool_use", "additionalModelResponseFields": self.fields}}}
        yield {"output": output_model(**self.values)}


def _recorded(name, **overrides):
    recorded = _fixture(name)
    return _LogprobModel({**recorded["tool_input"], **overrides}, recorded["content"])


def _softmax(logits):
    total = sum(math.exp(value) for value in logits.values())
    return {key: math.exp(value) / total for key, value in logits.items()}


def _tokens(*pieces):
    """Generated tokens; a piece is ``text`` or ``(text, [(alternative, logprob), ...])``."""
    tokens = []
    for piece in pieces:
        text, top = piece if isinstance(piece, tuple) else (piece, [(piece, 0.0)])
        tokens.append(
            {"token": text, "logprob": top[0][1], "top_logprobs": [{"token": t, "logprob": p} for t, p in top]}
        )
    return tokens


@pytest.mark.asyncio
async def test_logits_fill_every_answer_and_probabilities_are_their_softmax():
    engine = LLMDecisionModel(_recorded("qwen3_32b"))

    tru_decision = await engine.decide(Ticket, state="parcel a week late, refund or I cancel")

    assert (tru_decision.output.dept, tru_decision.output.cancel) == ("shipping", True)
    assert tru_decision.output.anger == pytest.approx(2.77, abs=0.01)  # probability-weighted level, not the argmax
    dept, anger, cancel = (tru_decision.answers[name] for name in ("dept", "anger", "cancel"))
    assert isinstance(dept, ChoiceAnswer) and isinstance(anger, ScoreAnswer) and isinstance(cancel, YesNoAnswer)
    assert set(dept.logits) == {"billing", "shipping", "technical"}
    assert set(anger.logits) == {0, 1, 2, 3, 4}
    assert set(cancel.logits) == {True, False}
    for answer in (dept, anger):
        assert answer.probabilities == pytest.approx(_softmax(answer.logits))
        assert answer.confidence is None  # not calibrated without temperature=
    assert dept.probabilities == pytest.approx({"billing": 0.2223, "shipping": 0.7760, "technical": 0.0017}, abs=1e-4)
    assert anger.probabilities[3] == pytest.approx(0.7567, abs=1e-4)
    assert cancel.probability == pytest.approx(0.9994, abs=1e-4)
    assert cancel.confidence is None
    assert dept.extras["label_mass"] == pytest.approx(1.0, abs=1e-3)
    assert engine.calibrated is False


@pytest.mark.asyncio
async def test_first_token_keys_a_multi_token_label():
    # Ministral spells "billing" as "b" + "illing": the option is keyed by its first token.
    tru_response = await LLMDecisionModel(_recorded("ministral_3_8b")).ask(
        "s",
        {
            "dept": Choice("?", options=dict.fromkeys(["billing", "shipping", "technical"])),
            "anger": Score("?", levels=["0", "1", "2", "3", "4"]),
            "cancel": YesNo("?"),
        },
    )

    dept = tru_response.answers["dept"]
    assert dept.choice == "billing"
    assert dept.probabilities == pytest.approx({"billing": 0.9241, "shipping": 0.0759, "technical": 0.0}, abs=1e-4)
    assert tru_response.answers["anger"].probabilities[4] == pytest.approx(0.7865, abs=1e-4)
    assert tru_response.answers["cancel"].extras["label_mass"] == pytest.approx(0.9997, abs=1e-4)


@pytest.mark.asyncio
async def test_temperature_makes_it_calibrated_and_rescales_the_logits():
    base = await LLMDecisionModel(_recorded("qwen3_32b")).decide(Ticket, state="s")
    engine = LLMDecisionModel(_recorded("qwen3_32b"), temperature=2.0)

    tru_decision = await engine.decide(Ticket, state="s")

    dept = tru_decision.answers["dept"]
    assert engine.calibrated is True
    assert dept.logits == base.answers["dept"].logits  # logits are pre-temperature
    assert dept.probabilities == pytest.approx(_softmax({k: v / 2.0 for k, v in dept.logits.items()}))
    assert dept.confidence == pytest.approx(max(dept.probabilities.values()))
    cancel = tru_decision.answers["cancel"]
    assert cancel.confidence == pytest.approx(abs(2 * cancel.probability - 1))
    assert engine.get_config()["temperature"] == 2.0


def test_min_confidence_gates_accept_it_only_with_temperature():
    with pytest.raises(ValueError, match="needs a calibrated DecisionModel"):
        DecisionStrategy(LLMDecisionModel(_recorded("qwen3_32b")), min_confidence=0.7)

    DecisionStrategy(LLMDecisionModel(_recorded("qwen3_32b"), temperature=1.3), min_confidence=0.7)


@pytest.mark.asyncio
async def test_fit_temperature_uses_the_logprob_logits():
    engine = LLMDecisionModel(_recorded("qwen3_32b"))
    examples = [("s", {"dept": "shipping"})] * 3 + [("s", {"dept": "billing"})]

    fit = await fit_temperature(engine, Ticket, examples)

    assert fit.pairs == 4
    assert 0.05 < float(fit) < 20.0
    engine.update_config(temperature=float(fit))
    assert engine.calibrated is True
    engine.update_config(temperature=None)
    assert engine.calibrated is False


@pytest.mark.parametrize("temperature", [0, -1.0, float("nan"), True, "1"])
def test_temperature_must_be_a_finite_positive_number(temperature):
    with pytest.raises(ValueError, match="temperature must be"):
        LLMDecisionModel(_recorded("qwen3_32b"), temperature=temperature)
    engine = LLMDecisionModel(_recorded("qwen3_32b"))
    with pytest.raises(ValueError, match="temperature must be"):
        engine.update_config(temperature=temperature)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        None,
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"logprobs": None}]},
        {"choices": [{"logprobs": {"content": []}}]},
        {"choices": [{"logprobs": {"content": [{"logprob": -0.1}]}}]},
        "not-a-mapping",
    ],
)
async def test_absent_or_malformed_logprobs_answer_one_hot(fields):
    llm = _LogprobModel({"q_ok": True, "q_team": "billing"}, [], fields=fields)

    tru_response = await LLMDecisionModel(llm).ask(
        "s", {"ok": YesNo("?"), "team": Choice("?", options=dict.fromkeys(["billing", "technical"]))}
    )

    assert tru_response.answers["ok"] == YesNoAnswer(probability=1.0)
    assert tru_response.answers["team"] == ChoiceAnswer("billing", {"billing": 1.0, "technical": 0.0})


@pytest.mark.asyncio
async def test_no_logprob_event_at_all_answers_one_hot_like_before():
    class _Plain(MockedModelProvider):
        async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
            yield {"event": {"messageStop": {"stopReason": "tool_use"}}}
            yield {"output": output_model(q_level=1)}

    tru_response = await LLMDecisionModel(_Plain([])).ask("s", {"level": Score("?", levels=["a", "b"])})

    assert tru_response.answers["level"] == ScoreAnswer(score=1.0, probabilities={0: 0.0, 1: 1.0})


@pytest.mark.asyncio
async def test_unreadable_field_answers_one_hot_and_records_why():
    shared = _tokens('{"', "q_team", '":', ' "', ("bill", [("bill", -0.1), ("tech", -2.4)]), 'ing"}')
    llm = _LogprobModel({"q_team": "billing", "q_ok": False}, shared)

    tru_response = await LLMDecisionModel(llm).ask(
        "s", {"team": Choice("?", options=dict.fromkeys(["billing", "billboard"])), "ok": YesNo("?")}
    )

    team = tru_response.answers["team"]
    assert team.probabilities == {"billing": 1.0, "billboard": 0.0}
    assert team.logits is None
    assert team.extras == {"logprobs_unavailable": "shared_first_token"}
    assert tru_response.answers["ok"].extras == {"logprobs_unavailable": "field_not_found"}
    assert tru_response.answers["ok"].probability == 0.0


def test_value_token_is_the_first_one_after_the_field_key():
    tokens = _tokens(
        '{"', "q", "_team", '":', ' "', ("ship", [("ship", -0.2), ("bill", -1.8), (" ship", -3.0)]), 'ping"}'
    )

    read = label_logits(tokens, "q_team", Choice("?", options=dict.fromkeys(["billing", "shipping"])), "shipping")

    assert read.logits == pytest.approx({"shipping": math.log(math.exp(-0.2) + math.exp(-3.0)), "billing": -1.8})
    assert read.label_mass == pytest.approx(math.exp(-0.2) + math.exp(-3.0) + math.exp(-1.8))


def test_value_token_inside_the_colon_token_is_found():
    # Some tokenizers fuse the colon and the value: '":"' or '": true'.
    tokens = _tokens('{"q_ok', ('": true', [('": true', -0.1), ('": false', -2.5)]), "}")

    read = label_logits(tokens, "q_ok", YesNo("?"), True)

    assert read.logits == {True: -0.1, False: -2.5}


def test_the_last_occurrence_of_the_field_key_is_used():
    # The field name can also appear earlier (for example inside a reasoning channel).
    tokens = _tokens("q_ok", '":', " maybe", ' {"q_ok":', (" false", [(" false", -0.3), (" true", -1.4)]), "}")

    read = label_logits(tokens, "q_ok", YesNo("?"), False)

    assert read.logits == {False: -0.3, True: -1.4}


@pytest.mark.parametrize(
    ("tokens", "question", "value", "reason"),
    [
        (
            _tokens('{"q_x":', ' "', ("c", [("c", -0.1), ("b", -2.0)]), '"}'),
            Choice("?", options=dict.fromkeys(["a", "b", "c"])),
            "c",
            "label_outside_top_k",
        ),
        (
            _tokens('{"q_x":', ' "', ("b", [("b", -0.1), ("a", -2.0)]), '"}'),
            Choice("?", options=dict.fromkeys(["a", "b"])),
            "a",
            "value_mismatch",
        ),
        (
            _tokens('{"q_x":', " ", ("2", [("2", -0.1), ("1", -2.0), ("0", -5.0)]), "}"),
            Score("?", levels=["a", "b", "c"]),
            2,
            None,
        ),
        (
            _tokens('{"q_x":', " ", ("1", [("1", -0.1), ("10", -2.0)])),
            Score("?", levels=[str(i) for i in range(11)]),
            1,
            "shared_first_token",
        ),
        (_tokens('{"q_x":', "   "), YesNo("?"), True, "field_not_found"),
        (_tokens('{"q_y": true}'), YesNo("?"), True, "field_not_found"),
    ],
)
def test_unreadable_value_tokens_are_reported_not_raised(tokens, question, value, reason):
    read = label_logits(tokens, "q_x", question, value)

    assert read == reason if reason else set(read.logits) == {0, 1, 2}


def test_an_alternative_that_begins_two_labels_refuses_the_logits():
    tokens = _tokens('{"q_x":', ' "', ("ship", [("ship", -0.1), ("bill", -2.0)]), 'ping"}')
    question = Choice("?", options=dict.fromkeys(["billing", "billboard", "shipping"]))

    assert label_logits(tokens, "q_x", question, "shipping") == "shared_first_token"


def test_forbidden_and_malformed_alternatives_are_ignored():
    tokens = _tokens('{"q_ok":', (" true", [(" true", -0.1), (" false", -9999.0), (" True", -3.0)]), "}")
    tokens[1]["top_logprobs"] += [{"token": " false", "logprob": "x"}, {"token": 5, "logprob": -1.0}, "junk"]

    assert label_logits(tokens, "q_ok", YesNo("?"), True) == "label_outside_top_k"

    tokens[1]["top_logprobs"].append({"token": "false", "logprob": -4.0})
    read = label_logits(tokens, "q_ok", YesNo("?"), True)
    assert read.logits == pytest.approx({True: math.log(math.exp(-0.1) + math.exp(-3.0)), False: -4.0})


def test_a_token_without_top_logprobs_uses_its_own_logprob():
    tokens = [{"token": '{"q_ok":'}, {"token": " true", "logprob": -0.01}]

    assert label_logits(tokens, "q_ok", YesNo("?"), True) == "label_outside_top_k"


def test_logprob_tokens_reads_only_a_converse_message_stop():
    content = [{"token": "a", "logprob": -0.1}]
    event = {
        "event": {"messageStop": {"additionalModelResponseFields": {"choices": [{"logprobs": {"content": content}}]}}}
    }

    assert logprob_tokens(event) == content
    assert logprob_tokens({"event": {"contentBlockStop": {}}}) is None
    assert logprob_tokens({"output": object()}) is None
    assert logprob_tokens({"event": "x"}) is None


@pytest.mark.asyncio
async def test_bedrock_model_non_streaming_passes_logprobs_through_to_the_answers():
    recorded = _fixture("qwen3_32b")
    with unittest.mock.patch.object(strands.models.bedrock.boto3, "Session") as session_cls:
        client = session_cls.return_value.client.return_value
        client.meta.region_name = "us-west-2"
        client.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "t1", "name": "DecisionAnswers", "input": recorded["tool_input"]}}
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 384, "outputTokens": 34, "totalTokens": 418},
            "metrics": {"latencyMs": 308},
            "additionalModelResponseFields": {"choices": [{"logprobs": {"content": recorded["content"]}}]},
        }
        bedrock = BedrockModel(
            model_id="qwen.qwen3-32b-v1:0",
            streaming=False,
            additional_request_fields={"logprobs": True, "top_logprobs": 20},
            additional_response_field_paths=["/choices/0/logprobs"],
        )

        tru_decision = await LLMDecisionModel(bedrock).decide(Ticket, state="parcel late")

    request = client.converse.call_args.kwargs
    assert request["additionalModelRequestFields"] == {"logprobs": True, "top_logprobs": 20}
    assert request["additionalModelResponseFieldPaths"] == ["/choices/0/logprobs"]
    assert tru_decision.answers["dept"].probabilities["shipping"] == pytest.approx(0.7760, abs=1e-4)
    assert tru_decision.usage["outputTokens"] == 34
