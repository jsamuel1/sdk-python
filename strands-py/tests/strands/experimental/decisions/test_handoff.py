import json
import logging
import shutil
import tempfile

import pytest

from strands import Agent
from strands.experimental.decisions import (
    COMPLETE_OPTION,
    DECISION_STATE_KEY,
    Decision,
    DecisionHandoffStrategy,
    HandoffDecision,
    LLMDecisionModel,
)
from strands.hooks import AfterNodeCallEvent
from strands.multiagent import Handoff, Status, Swarm
from strands.session.file_session_manager import FileSessionManager
from tests.fixtures.mocked_decision_model import MockedDecisionModel, choice
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _text(text):
    return {"role": "assistant", "content": [{"text": text}]}


def _handoff_call(agent_name):
    tool_input = {"agent_name": agent_name, "message": f"over to {agent_name}"}
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "h1", "name": "handoff_to_agent", "input": tool_input}}],
    }


def _agent(name, *responses, description=None):
    return Agent(
        name=name,
        description=description,
        model=MockedModelProvider(list(responses)),
        system_prompt=f"You are {name}",
        callback_handler=None,
    )


def _team(triage_turns=("Triaged: a refund request",), billing_turns=("Refund issued",), tech_turns=("Fixed",)):
    return [
        _agent("triage", *[_text(t) if isinstance(t, str) else t for t in triage_turns], description="Front line"),
        _agent("billing", *[_text(t) for t in billing_turns], description="Charges and refunds"),
        _agent("technical", *[_text(t) for t in tech_turns], description="Bugs and outages"),
    ]


def _decision(result, node_id):
    return result.results[node_id].result.state.get(DECISION_STATE_KEY)


@pytest.mark.asyncio
async def test_strategy_hands_off_when_node_does_not():
    decisions = MockedDecisionModel(
        {"next": choice("c0", {"c0": 0.8, "c1": 0.1, COMPLETE_OPTION: 0.1}, 0.8)},
        {"next": choice(COMPLETE_OPTION, {"c0": 0.05, "c1": 0.05, COMPLETE_OPTION: 0.9}, 0.9)},
    )
    swarm = Swarm(_team(), handoff_strategy=DecisionHandoffStrategy(decisions, min_confidence=0.7))

    result = await swarm.invoke_async("I was charged twice")

    assert result.status == Status.COMPLETED
    assert [node.node_id for node in result.node_history] == ["triage", "billing"]
    state, questions = decisions.requests[0]
    assert (state["agent_instructions"], state["response"]) == ("You are triage", "Triaged: a refund request")
    assert "User Request: I was charged twice" in state["request"]
    options = questions["next"].options
    assert list(options) == ["c0", "c1", COMPLETE_OPTION]
    tru_evidence = [json.loads(options[key]) for key in ("c0", "c1")]
    exp_evidence = [
        {"name": "billing", "description": "Charges and refunds"},
        {"name": "technical", "description": "Bugs and outages"},
    ]
    assert tru_evidence == exp_evidence
    assert "no other agent should act" in options[COMPLETE_OPTION]
    billing_input = swarm.nodes["billing"].executor.messages[0]["content"][0]["text"]
    assert "a decision model routed the task to you (confidence 0.80)" in billing_input
    assert "Triaged: a refund request" in billing_input


@pytest.mark.asyncio
async def test_decision_recorded_on_the_node_that_ran_keyed_by_node_id():
    decisions = MockedDecisionModel(
        {"next": choice("c1", {"c0": 0.2, "c1": 0.7, COMPLETE_OPTION: 0.1}, 0.7)},
        {"next": choice(COMPLETE_OPTION, {"c0": 0.1, "c1": 0.1, COMPLETE_OPTION: 0.8}, 0.8)},
    )
    swarm = Swarm(_team(), handoff_strategy=DecisionHandoffStrategy(decisions))

    result = await swarm.invoke_async("The export button 500s")

    triage_decision = _decision(result, "triage")
    assert isinstance(triage_decision, Decision)
    assert triage_decision.output == HandoffDecision(next="technical")
    tru_answer = triage_decision.answers["next"]
    assert (tru_answer.choice, dict(tru_answer.probabilities), tru_answer.confidence) == (
        "technical",
        {"billing": 0.2, "technical": 0.7, COMPLETE_OPTION: 0.1},
        0.7,
    )
    assert triage_decision.model_id == "mock-s1-1.0"
    assert _decision(result, "technical").output == HandoffDecision(next=COMPLETE_OPTION)


@pytest.mark.asyncio
async def test_explicit_handoff_tool_call_wins_and_strategy_is_not_asked():
    decisions = MockedDecisionModel({"next": choice(COMPLETE_OPTION, confidence=0.9)})
    team = _team(triage_turns=(_handoff_call("technical"), "handed off"))
    swarm = Swarm(team, handoff_strategy=DecisionHandoffStrategy(decisions, min_confidence=0.5))

    result = await swarm.invoke_async("The export button 500s")

    assert [node.node_id for node in result.node_history] == ["triage", "technical"]
    tru_asked_about = [state["agent_instructions"] for state, _ in decisions.requests]
    assert tru_asked_about == ["You are technical"]
    assert DECISION_STATE_KEY not in result.results["triage"].result.state


@pytest.mark.asyncio
async def test_complete_answer_ends_the_swarm():
    decisions = MockedDecisionModel({"next": choice(COMPLETE_OPTION, confidence=0.95)})
    swarm = Swarm(_team(), handoff_strategy=DecisionHandoffStrategy(decisions, min_confidence=0.5))

    result = await swarm.invoke_async("Thanks, all sorted")

    assert result.status == Status.COMPLETED
    assert [node.node_id for node in result.node_history] == ["triage"]


@pytest.mark.parametrize(
    ("fallback", "exp_history"),
    [("complete", ["triage"]), ("stay", ["triage", "triage"])],
)
@pytest.mark.asyncio
async def test_below_floor_uses_fallback_and_still_records_decision(fallback, exp_history):
    decisions = MockedDecisionModel(
        {"next": choice("c0", {"c0": 0.4, "c1": 0.35, COMPLETE_OPTION: 0.25}, 0.4)},
        {"next": choice(COMPLETE_OPTION, confidence=0.9)},
    )
    team = _team(triage_turns=("Not sure yet", "Resolved it myself"))
    swarm = Swarm(team, handoff_strategy=DecisionHandoffStrategy(decisions, min_confidence=0.7, fallback=fallback))

    result = await swarm.invoke_async("Something is off")

    assert [node.node_id for node in result.node_history] == exp_history
    assert result.status == Status.COMPLETED
    if fallback == "stay":
        rerun_input = swarm.nodes["triage"].executor.messages[0]["content"][0]["text"]
        assert "could not choose the next agent with enough confidence" in rerun_input
    else:
        assert _decision(result, "triage").answers["next"].confidence == 0.4


@pytest.mark.parametrize(("fallback", "exp_history"), [("complete", ["triage"]), ("stay", ["triage", "triage"])])
@pytest.mark.asyncio
async def test_decision_error_uses_fallback_and_logs_warning(fallback, exp_history, caplog):
    decisions = MockedDecisionModel(RuntimeError("service down"), {"next": choice(COMPLETE_OPTION, confidence=0.9)})
    team = _team(triage_turns=("First pass", "Second pass"))
    swarm = Swarm(team, handoff_strategy=DecisionHandoffStrategy(decisions, min_confidence=0.7, fallback=fallback))

    with caplog.at_level(logging.WARNING, logger="strands.experimental.decisions._handoff"):
        result = await swarm.invoke_async("Something is off")

    assert result.status == Status.COMPLETED
    assert [node.node_id for node in result.node_history] == exp_history
    assert "error_type=<RuntimeError> | handoff decision failed, using fallback" in caplog.text


@pytest.mark.asyncio
async def test_stay_fallback_is_bounded_by_max_handoffs():
    decisions = MockedDecisionModel(*[{"next": choice("c0", confidence=0.1)} for _ in range(5)])
    team = _team(triage_turns=tuple(f"pass {index}" for index in range(5)))
    swarm = Swarm(
        team,
        max_handoffs=3,
        handoff_strategy=DecisionHandoffStrategy(decisions, min_confidence=0.7, fallback="stay"),
    )

    result = await swarm.invoke_async("Loop forever")

    assert result.status == Status.FAILED
    assert [node.node_id for node in result.node_history] == ["triage", "triage", "triage"]


@pytest.mark.asyncio
async def test_strategy_handoffs_trip_repetitive_handoff_detection():
    # triage -> billing -> triage -> billing: two unique nodes in a window of four.
    decisions = MockedDecisionModel(
        {"next": choice("c0", confidence=0.9)},  # triage: candidates billing(c0), technical(c1)
        {"next": choice("c0", confidence=0.9)},  # billing: candidates triage(c0), technical(c1)
        {"next": choice("c0", confidence=0.9)},
        {"next": choice("c0", confidence=0.9)},
    )
    team = _team(triage_turns=("t1", "t2"), billing_turns=("b1", "b2"))
    swarm = Swarm(
        team,
        repetitive_handoff_detection_window=4,
        repetitive_handoff_min_unique_agents=3,
        handoff_strategy=DecisionHandoffStrategy(decisions),
    )

    result = await swarm.invoke_async("Ping pong")

    assert result.status == Status.FAILED
    assert [node.node_id for node in result.node_history] == ["triage", "billing", "triage", "billing"]


@pytest.mark.asyncio
async def test_strategy_handoff_is_checkpointed_and_resumes_at_the_target():
    decisions = MockedDecisionModel(
        {"next": choice("c0", confidence=0.9)}, {"next": choice(COMPLETE_OPTION, confidence=0.9)}
    )
    with tempfile.TemporaryDirectory() as live_dir, tempfile.TemporaryDirectory() as crash_dir:
        swarm = Swarm(
            _team(),
            session_manager=FileSessionManager(session_id="handoff", storage_dir=live_dir),
            handoff_strategy=DecisionHandoffStrategy(decisions),
        )
        captured = []

        def capture_after_first_node(event):
            if not captured:
                shutil.rmtree(crash_dir, ignore_errors=True)
                shutil.copytree(live_dir, crash_dir)
                captured.append(swarm.serialize_state())

        swarm.hooks.add_callback(AfterNodeCallEvent, capture_after_first_node, order=1000)
        await swarm.invoke_async("I was charged twice")

        assert captured[0]["next_nodes_to_execute"] == ["billing"]

        resumed = Swarm(
            _team(billing_turns=("Refund issued after restart",)),
            session_manager=FileSessionManager(session_id="handoff", storage_dir=crash_dir),
            handoff_strategy=DecisionHandoffStrategy(MockedDecisionModel({"next": choice(COMPLETE_OPTION)})),
        )
        resumed_result = await resumed.invoke_async("I was charged twice")

        assert resumed_result.status == Status.COMPLETED
        assert [node.node_id for node in resumed_result.node_history] == ["triage", "billing"]
        assert "Refund issued after restart" in str(resumed_result.results["billing"].result)


@pytest.mark.asyncio
async def test_strategy_error_rolls_back_turn_and_fails_swarm():
    class Broken:
        async def select(self, context):
            raise RuntimeError("strategy bug")

    swarm = Swarm(_team(), handoff_strategy=Broken())

    result = await swarm.invoke_async("anything")

    assert result.status == Status.FAILED
    assert swarm.state.handoff_node is None


@pytest.mark.asyncio
async def test_custom_strategy_receives_context_and_foreign_node_is_refused():
    seen = []

    class Picks:
        def __init__(self, node):
            self.node = node

        async def select(self, context):
            seen.append(
                (context.current.node_id, [n.node_id for n in context.candidates], [n.node_id for n in context.history])
            )
            return Handoff(node=self.node, message="go")

    team = _team()
    other = Swarm(_team())
    swarm = Swarm(team, handoff_strategy=Picks(other.nodes["billing"]))

    result = await swarm.invoke_async("anything")

    assert seen == [("triage", ["billing", "technical"], ["triage"])]
    assert result.status == Status.FAILED


def test_rejects_min_confidence_on_uncalibrated_model():
    with pytest.raises(ValueError, match=r"DecisionHandoffStrategy\(min_confidence=0.7\) needs a calibrated"):
        DecisionHandoffStrategy(LLMDecisionModel(MockedModelProvider([])), min_confidence=0.7)


def test_rejects_unknown_fallback_and_bad_budget():
    with pytest.raises(ValueError, match="fallback must be 'complete' or 'stay'"):
        DecisionHandoffStrategy(MockedDecisionModel(), fallback="guess")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token budgets must be greater than zero"):
        DecisionHandoffStrategy(MockedDecisionModel(), max_response_tokens=0)


@pytest.mark.asyncio
async def test_node_named_complete_is_refused_at_decision_time():
    team = [_agent("triage", _text("done")), _agent(COMPLETE_OPTION, _text("x"))]
    swarm = Swarm(team, handoff_strategy=DecisionHandoffStrategy(MockedDecisionModel()))

    result = await swarm.invoke_async("anything")

    assert result.status == Status.FAILED


@pytest.mark.asyncio
async def test_without_strategy_a_node_that_does_not_hand_off_completes():
    result = await Swarm(_team()).invoke_async("anything")

    assert [node.node_id for node in result.node_history] == ["triage"]
    assert DECISION_STATE_KEY not in result.results["triage"].result.state
