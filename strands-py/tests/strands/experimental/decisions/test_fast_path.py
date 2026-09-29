import unittest.mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from strands import Agent, tool
from strands.event_loop._synthesized import synthesized_source
from strands.experimental.decisions import DecisionResponse, FastPath, LLMDecisionModel, ToolCall
from strands.experimental.decisions._fast_path import _latest_tool_result_text
from strands.hooks import AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent
from strands.session.repository_session_manager import RepositorySessionManager
from strands.telemetry.tracer import Tracer
from tests.fixtures.mock_session_repository import MockedSessionRepository
from tests.fixtures.mocked_decision_model import MockedDecisionModel, choice
from tests.fixtures.mocked_model_provider import MockedModelProvider

CLICKS: list[str] = []


@tool
def click(selector: str) -> str:
    """Click an element."""
    CLICKS.append(selector)
    return f"clicked {selector}"


@tool
def scroll(dy: int) -> str:
    """Scroll the page."""
    return f"scrolled {dy}"


ACTIONS = {
    "click_submit": ToolCall("click", {"selector": "#submit"}),
    "scroll_down": ToolCall("scroll", {"dy": 600}, description="Scroll one screen down"),
}
OTHER = {"action": choice("other", {"click_submit": 0.1, "scroll_down": 0.1, "other": 0.8}, 0.8)}


def _served(option="click_submit", confidence=0.95):
    return {"action": choice(option, {"click_submit": 0.0, "scroll_down": 0.0, "other": 0.0, option: 1.0}, confidence)}


def _llm(*texts):
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]} for text in texts])


def _agent(decisions, llm, **fast_path_options):
    fast_path_options.setdefault("min_confidence", 0.8)
    return Agent(
        model=llm,
        tools=[click, scroll],
        plugins=[FastPath(decisions, ACTIONS, **fast_path_options)],
        callback_handler=None,
    )


@pytest.fixture(autouse=True)
def _reset_clicks():
    CLICKS.clear()


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = Tracer()
    tracer.tracer_provider = provider
    tracer.tracer = provider.get_tracer(tracer.service_name)
    with unittest.mock.patch("strands.telemetry.tracer._tracer_instance", tracer):
        yield exporter


def test_confident_answer_serves_the_step_and_the_llm_resumes():
    decisions = MockedDecisionModel(_served(), OTHER)
    llm = _llm("submitted")
    agent = _agent(decisions, llm)

    result = agent("Submit the form")

    assert str(result).strip() == "submitted"
    assert CLICKS == ["#submit"]
    assert llm.index == 1
    synthesized, tool_result = agent.messages[1], agent.messages[2]
    tool_use = synthesized["content"][0]["toolUse"]
    assert tool_use["name"] == "click"
    assert tool_use["input"] == {"selector": "#submit"}
    assert tool_use["toolUseId"].startswith("s1-")
    assert synthesized["metadata"] == {
        "custom": {"strands": {"source": "decision", "decision_model": "mock-s1-1.0", "confidence": 0.95}}
    }
    assert tool_result["content"][0]["toolResult"]["toolUseId"] == tool_use["toolUseId"]


@pytest.mark.parametrize("model_id", ["mock-s1", None])
def test_attribution_falls_back_when_the_response_names_no_model(model_id):
    class Anonymous(MockedDecisionModel):
        async def _ask(self, state, questions, **kwargs):
            response = await super()._ask(state, questions, **kwargs)
            return DecisionResponse(answers=response.answers, usage=response.usage)

    decisions = Anonymous(_served(), OTHER, model_id=model_id)
    agent = _agent(decisions, _llm("done"))

    agent("Submit the form")

    assert agent.messages[1]["metadata"]["custom"]["strands"]["decision_model"] == model_id


def test_question_offers_every_action_plus_other_and_projects_the_latest_tool_result():
    decisions = MockedDecisionModel(_served(), OTHER)
    agent = _agent(decisions, _llm("done"), instructions="Next action?")

    agent("Submit the form")

    first_state, questions = decisions.requests[0]
    assert first_state == {"request": "Submit the form", "agent_instructions": ""}
    assert questions["action"].instructions == "Next action?"
    assert questions["action"].options == {
        "click_submit": '{"tool": "click", "input": {"selector": "#submit"}}',
        "scroll_down": "Scroll one screen down",
        "other": "None of the actions fits; let the reasoning model decide",
    }
    second_state, _ = decisions.requests[1]
    assert second_state["latest_tool_result"] == "clicked #submit"


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(OTHER, id="other"),
        pytest.param(_served(confidence=0.79), id="below-floor"),
        pytest.param(_served(confidence=None), id="no-confidence"),
        pytest.param(RuntimeError("service down"), id="decision-error"),
    ],
)
def test_pass_through_runs_the_llm_unchanged(answer):
    decisions = MockedDecisionModel(answer)
    llm = _llm("llm answer")
    agent = _agent(decisions, llm)

    result = agent("Submit the form")

    assert str(result).strip() == "llm answer"
    assert CLICKS == []
    assert "metadata" not in agent.messages[1] or "custom" not in agent.messages[1]["metadata"]


def test_decision_error_is_logged(caplog):
    agent = _agent(MockedDecisionModel(RuntimeError("secret url")), _llm("llm answer"))

    with caplog.at_level("WARNING"):
        agent("Submit the form")

    assert "reason=<decision_error>, error_type=<RuntimeError>" in caplog.text
    assert "secret url" not in caplog.text


def test_action_whose_tool_is_not_offered_passes_through():
    decisions = MockedDecisionModel(_served())
    llm = _llm("llm answer")
    agent = Agent(model=llm, tools=[scroll], plugins=[FastPath(decisions, ACTIONS, min_confidence=0.8)])

    assert str(agent("Submit the form")).strip() == "llm answer"


def test_max_consecutive_forces_an_llm_turn():
    decisions = MockedDecisionModel(_served(), _served(), _served())
    llm = _llm("llm turn")
    agent = _agent(decisions, llm, max_consecutive=2)

    agent("Keep submitting")

    assert CLICKS == ["#submit", "#submit"]
    assert len(decisions.requests) == 2
    assert llm.index == 1


def test_a_model_turn_resets_the_consecutive_count():
    decisions = MockedDecisionModel(_served(), _served())
    llm = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "scroll", "input": {"dy": 1}}}]},
            {"role": "assistant", "content": [{"text": "done"}]},
        ]
    )
    agent = _agent(decisions, llm, max_consecutive=1)

    agent("Submit, then scroll, then submit")

    assert CLICKS == ["#submit", "#submit"]
    # served, forced model turn, served again after the model turn, forced again
    assert len(decisions.requests) == 2
    assert llm.index == 2


def test_forced_tool_choice_skips_the_fast_path():
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    decisions = MockedDecisionModel(OTHER, _served())
    llm = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "no tool"}]},
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "so", "name": "Answer", "input": {"text": "ok"}}}],
            },
        ]
    )
    agent = _agent(decisions, llm)

    result = agent("Answer", structured_output_model=Answer)

    assert result.structured_output.text == "ok"
    assert len(decisions.requests) == 1


def test_synthesized_turn_fires_no_after_model_call_event_but_tool_hooks_run():
    decisions = MockedDecisionModel(_served(), OTHER)
    agent = _agent(decisions, _llm("done"))
    before, after, tools = [], [], []
    agent.add_hook(lambda event: before.append(event), BeforeModelCallEvent)
    agent.add_hook(lambda event: after.append(event), AfterModelCallEvent)
    agent.add_hook(lambda event: tools.append(event.tool_use["name"]), AfterToolCallEvent)

    agent("Submit the form")

    assert len(before) == 2
    assert [event.stop_response.message["content"][0]["text"] for event in after] == ["done"]
    assert tools == ["click"]


def test_only_decision_usage_is_billed():
    decisions = MockedDecisionModel(_served(), _served(), OTHER)
    llm = MockedModelProvider(
        [{"role": "assistant", "content": [{"text": "done"}]}],
        usages=[{"inputTokens": 100, "outputTokens": 5, "totalTokens": 105}],
    )
    agent = _agent(decisions, llm)

    result = agent("Submit twice")

    assert result.metrics.accumulated_usage["totalTokens"] == 105
    assert "usage" not in agent.messages[1].get("metadata", {})


def test_decision_span_replaces_the_chat_span(spans):
    agent = _agent(MockedDecisionModel(_served(), OTHER), _llm("done"))

    agent("Submit the form")

    finished = spans.get_finished_spans()
    cycles = [span for span in finished if span.name == "execute_event_loop_cycle"]
    decisions = [span for span in finished if span.name == "decision"]
    chats = [span for span in finished if span.name == "chat"]
    assert len(cycles) == 2 and len(decisions) == 2 and len(chats) == 1
    served, passed = decisions
    assert served.attributes["strands.fast_path.served"] is True
    assert served.attributes["strands.fast_path.action"] == "click_submit"
    assert served.attributes["strands.source"] == "decision"
    assert served.parent.span_id == cycles[0].context.span_id
    assert not any(chat.parent.span_id == cycles[0].context.span_id for chat in chats)
    assert passed.attributes["strands.fast_path.served"] is False
    assert passed.attributes["strands.fast_path.reason"] == "other"
    assert "strands.fast_path.action" not in passed.attributes


def test_session_round_trip_preserves_the_marking():
    repository = MockedSessionRepository()
    agent = Agent(
        agent_id="browser",
        model=_llm("done"),
        tools=[click, scroll],
        plugins=[FastPath(MockedDecisionModel(_served(), OTHER), ACTIONS, min_confidence=0.8)],
        session_manager=RepositorySessionManager(session_id="s", session_repository=repository),
        callback_handler=None,
    )
    agent("Submit the form")
    tool_use_id = agent.messages[1]["content"][0]["toolUse"]["toolUseId"]

    restored = Agent(
        agent_id="browser",
        model=_llm("unused"),
        session_manager=RepositorySessionManager(session_id="s", session_repository=repository),
        callback_handler=None,
    )

    synthesized = restored.messages[1]
    assert synthesized["content"][0]["toolUse"]["toolUseId"] == tool_use_id
    assert tool_use_id.startswith("s1-")
    assert synthesized["metadata"]["custom"]["strands"] == {
        "source": "decision",
        "decision_model": "mock-s1-1.0",
        "confidence": 0.95,
    }
    assert "custom" not in restored.messages[3].get("metadata", {})


def test_metadata_is_not_sent_to_the_model():
    decisions = MockedDecisionModel(_served(), OTHER)
    llm = _llm("done")
    agent = _agent(decisions, llm)
    with unittest.mock.patch.object(llm, "stream", wraps=llm.stream) as stream:
        agent("Submit the form")

    sent = stream.call_args.args[0]
    assert all("metadata" not in message for message in sent)


def test_uncalibrated_model_is_refused():
    with pytest.raises(ValueError, match=r"FastPath\(min_confidence=0.8\) needs a calibrated DecisionModel"):
        FastPath(LLMDecisionModel(_llm()), ACTIONS, min_confidence=0.8)


@pytest.mark.parametrize(
    ("actions", "options", "error", "match"),
    [
        ({}, {}, ValueError, "at least one action"),
        ({"other": ToolCall("click")}, {}, ValueError, "'other' is reserved"),
        ({"a": ("click", {})}, {}, TypeError, "must be a ToolCall"),
        (ACTIONS, {"min_confidence": 0.0}, ValueError, "min_confidence"),
        (ACTIONS, {"min_confidence": 1.5}, ValueError, "min_confidence"),
        (ACTIONS, {"max_consecutive": 0}, ValueError, "max_consecutive"),
        (ACTIONS, {"max_request_tokens": True}, ValueError, "max_request_tokens"),
    ],
)
def test_invalid_configuration_is_refused(actions, options, error, match):
    options.setdefault("min_confidence", 0.8)
    with pytest.raises(error, match=match):
        FastPath(MockedDecisionModel(), actions, **options)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, None),
        ({"custom": {"strands": "decision"}}, None),
        ({"custom": {"strands": {"source": 7}}}, None),
        ({"custom": {"strands": {"source": "decision"}}}, "decision"),
    ],
)
def test_synthesized_source_reads_only_a_well_formed_marker(metadata, expected):
    message = {"role": "assistant", "content": []}
    if metadata is not None:
        message["metadata"] = metadata

    assert synthesized_source(message) == expected


def test_error_tool_result_and_media_are_labelled_in_state():
    messages = [
        {"role": "user", "content": [{"text": "Submit"}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t0",
                        "status": "error",
                        "content": [{"json": {"code": 404}}, {"image": {"format": "png", "source": {"bytes": b""}}}],
                    }
                }
            ],
        },
    ]

    assert _latest_tool_result_text(messages, 1_000) == '[error]\n{"code": 404}\n[Image]'
    assert _latest_tool_result_text(messages[:1], 1_000) is None
    assert _latest_tool_result_text([], 1_000) is None
