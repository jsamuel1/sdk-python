from unittest import mock

import pytest

from strands.experimental.decisions import DecisionGuard, GuardPolicy, Score, YesNo, decision_classifier
from strands.interventions.actions import Confirm, Deny, Guide, Proceed
from tests.fixtures.mocked_decision_model import MockedDecisionModel, score, yes


def _event(name="shell", tool_input=None):
    event = mock.Mock()
    event.tool_use = {"toolUseId": "t1", "name": name, "input": tool_input or {"command": "rm -rf /data"}}
    return event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.97, Deny), (0.6, Confirm), (0.2, Proceed)],
    ids=["deny", "confirm", "proceed"],
)
async def test_before_tool_call_applies_floors(probability, expected):
    decisions = MockedDecisionModel({"needs_approval": yes(probability), "severity": score(2.9, levels=4)})

    tru_action = await DecisionGuard(decisions).before_tool_call(_event())

    assert isinstance(tru_action, expected)
    assert f"approval p={probability:.2f}, severity 2.9/3" in (tru_action.reason or "")
    state, questions = decisions.requests[0]
    assert state == {"tool": "shell", "input": {"command": "rm -rf /data"}}
    assert set(questions) == {"needs_approval", "severity"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [("confirm", Confirm), ("deny", Deny), ("proceed", Proceed)],
)
async def test_before_tool_call_on_decision_error(mode, expected, caplog):
    guard = DecisionGuard(MockedDecisionModel(RuntimeError("down")), on_decision_error=mode)

    tru_action = await guard.before_tool_call(_event())

    assert isinstance(tru_action, expected)
    assert "risk check unavailable (RuntimeError)" in (tru_action.reason or "")
    assert "guard decision failed" in caplog.text


@pytest.mark.parametrize(
    ("policy", "p", "severity", "expected"),
    [
        (GuardPolicy(), 0.97, 0.0, Deny),
        (GuardPolicy(deny_min_severity=2.5), 0.97, 2.9, Deny),
        (GuardPolicy(deny_min_severity=2.5), 0.97, 1.6, Confirm),  # severity gate blocks Deny -> fail-safe Confirm
        (GuardPolicy(), 0.6, 3.0, Confirm),
        (GuardPolicy(), 0.2, 3.0, Proceed),
        (GuardPolicy(deny_above=None), 1.0, 3.0, Confirm),  # never auto-deny
        (GuardPolicy(confirm_above=0.3, deny_above=0.8), 0.86, 1.6, Deny),
    ],
    ids=["deny", "deny_severe", "severity_gate_confirms", "confirm", "proceed", "no_auto_deny", "tight_floors"],
)
def test_guard_policy_action_evaluation_order(policy, p, severity, expected):
    assert policy.action(p, severity) is expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confirm_above": 0.9, "deny_above": 0.5}, "confirm_above <= deny_above"),
        ({"confirm_above": 1.5, "deny_above": None}, "confirm_above <= deny_above"),
        ({"deny_min_severity": -1}, "deny_min_severity must be >= 0"),
    ],
)
def test_guard_policy_validates(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GuardPolicy(**kwargs)


def test_init_rejects_severity_floor_above_the_scale():
    with pytest.raises(ValueError, match=r"shell: deny_min_severity must be <= 3"):
        DecisionGuard(MockedDecisionModel(), policies={"shell": GuardPolicy(deny_min_severity=3.5)})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "expected"),
    [("shell", Deny), ("send_email", Confirm), ("read_file", Confirm)],
    ids=["per_tool_deny", "per_tool_no_auto_deny", "default_policy"],
)
async def test_before_tool_call_uses_the_tool_policy(tool, expected):
    decisions = MockedDecisionModel({"needs_approval": yes(0.86), "severity": score(1.6, levels=4)})
    guard = DecisionGuard(
        decisions,
        policies={"shell": GuardPolicy(confirm_above=0.3, deny_above=0.8), "send_email": GuardPolicy(deny_above=None)},
    )

    assert isinstance(await guard.before_tool_call(_event(name=tool)), expected)


def test_guard_default_fails_closed():
    assert DecisionGuard(MockedDecisionModel())._on_decision_error == "confirm"


@pytest.mark.asyncio
async def test_decision_classifier_thresholds():
    classifier = decision_classifier(MockedDecisionModel({"needs_approval": yes(0.7)}, {"needs_approval": yes(0.3)}))

    first = await classifier(_event())
    second = await classifier(_event())

    assert (first.requires_human_in_the_loop, first.reason) == (True, "approval p=0.70")
    assert second.requires_human_in_the_loop is False


@pytest.mark.asyncio
async def test_decision_classifier_propagates_errors():
    classifier = decision_classifier(MockedDecisionModel(RuntimeError("down")))

    with pytest.raises(RuntimeError, match="down"):
        await classifier(_event())


def _model_event(text="The answer is 42."):
    event = mock.Mock()
    event.stop_response = mock.Mock(message={"role": "assistant", "content": [{"text": text}]})
    return event


_OUTPUT_QUESTIONS = {"unsupported": YesNo("Does `response` make a claim with no cited source?")}


@pytest.mark.asyncio
async def test_before_tool_call_unsure_answer_confirms():
    decisions = MockedDecisionModel({"needs_approval": yes(0.5), "severity": score(1.5, levels=4)})

    tru_action = await DecisionGuard(decisions).before_tool_call(_event())

    assert isinstance(tru_action, Confirm)


@pytest.mark.asyncio
async def test_after_model_call_guides_then_exhausts_budget():
    decisions = MockedDecisionModel({"unsupported": yes(0.8)}, {"unsupported": yes(0.9)})
    guard = DecisionGuard(decisions, output_questions=_OUTPUT_QUESTIONS, max_output_retries=1)

    first = await guard.after_model_call(_model_event())
    second = await guard.after_model_call(_model_event())
    guard.before_invocation(mock.Mock())
    third = await guard.after_model_call(_model_event())

    assert first == Guide(
        feedback="Revise your response; a check flagged it: Does `response` make a claim with no cited source?",
        reason="unsupported p=0.80",
    )
    assert second == Proceed(reason="output retry budget exhausted")
    assert isinstance(third, Guide)
    assert [state for state, _ in decisions.requests] == [{"response": "The answer is 42."}] * 2


@pytest.mark.asyncio
async def test_after_model_call_passes_and_skips():
    decisions = MockedDecisionModel({"unsupported": yes(0.1)})
    guard = DecisionGuard(decisions, output_questions=_OUTPUT_QUESTIONS)
    no_response = mock.Mock(stop_response=None)

    assert await guard.after_model_call(_model_event()) == Proceed()
    assert await guard.after_model_call(no_response) == Proceed()
    assert await DecisionGuard(decisions).after_model_call(_model_event()) == Proceed()
    assert len(decisions.requests) == 1


@pytest.mark.asyncio
async def test_after_model_call_error_proceeds(caplog):
    guard = DecisionGuard(MockedDecisionModel(RuntimeError("down")), output_questions=_OUTPUT_QUESTIONS)

    tru_action = await guard.after_model_call(_model_event())

    assert tru_action == Proceed(reason="output check unavailable (RuntimeError)")
    assert "guard output check failed" in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_guide_above": 1.5},
        {"max_output_retries": -1},
        {"output_questions": {"bad": Score("How bad?", levels=["ok", "bad"])}},
    ],
)
def test_init_validates_output_options(kwargs):
    with pytest.raises(ValueError, match="output"):
        DecisionGuard(MockedDecisionModel(), **kwargs)
