# System One Decision Models — P1: Swarm handoff and the fast path

**Status**: Proposed

**Date**: 2026-09-27

**Issue**: [#4551](https://github.com/strands-agents/harness-sdk/issues/4551)

**Parent**: [0020 System One Decision Models](./0020-system-one-decision-models.md). This addendum is the "own small design" that 0020 requires before two P1 adapters ship. It reuses 0020's vocabulary (`DecisionModel`, `Choice`, `YesNo`, `min_confidence`, `DECISION_STATE_KEY`) and changes none of the P0 API.

**Scope**: Python and TypeScript together. The TypeScript port of the P0 surface (`DecisionModel`, schemas, `TypeSafeDecisionModel`, `DecisionStrategy`) already exists, so neither adapter needs a staggered port.

## Problem

0020 ships decision adapters at every seam that already has a deterministic hook: routing (`RoutingStrategy`), graph edges (`EdgeCondition`), entry dispatch (`AgentBase`), tools, and tool gating (`InterventionHandler`). Two seams from the issue's use cases are left out, because each one makes a System One model do something that is currently an LLM's job:

1. **Swarm handoff.** A `Swarm` moves control only when the active agent's LLM calls `handoff_to_agent(agent_name, message, context)`. So routing between swarm members is a side effect of a generative turn. It carries no confidence, and nothing outside that turn can override it. Use case 1 (dynamic multi-agent routing) wants the handoff to be a calibrated decision.
2. **System-One-first action selection.** Use case 3 (browser use) runs an LLM turn on every step, even when the next action is one of a few closed options that a decision model answers in about 0.6 s at roughly 1/30 of the cost (baseline E1–E3). 0020 describes this "fast path" but defers it, because it writes turns into the transcript that no LLM produced.

A third P1 item needs no design, only a dependency. Usage accounting will route decision-call usage into `accumulated_usage` through the auxiliary-call runner once #4005 lands. Until then, decision usage lives on the decision span only (0020 §Observability).

## Goals

- **H1 Deterministic handoff.** A swarm can hand off on a decision model's answer with no LLM tool call, and with no change to the default LLM-driven behaviour.
- **H2 Uncertainty falls back.** Below the floor, control reverts to the existing LLM handoff tool. It is never a guess.
- **H3 Honest transcript.** A fast-path turn is recorded as synthesized: it is marked, attributed, and replayable, and never passed off as model output.
- **H4 No new agent-loop concepts.** Both adapters sit on existing seams (Swarm's transition point, `InvokeModelStage` middleware) and add no hook events.

## Design

### 1. Swarm `handoff_strategy=`

The P0 swarm transition has one writer: `Swarm._handle_handoff(target_node, message, context)`, called from the injected `handoff_to_agent` tool. P1 adds a second writer, called once after each node completes and before the swarm checks for a handoff.

```python
class HandoffStrategy(Protocol):
    async def select(self, context: HandoffContext) -> Handoff | None: ...

@dataclass(frozen=True)
class HandoffContext:
    current: SwarmNode
    candidates: Sequence[SwarmNode]          # every node except current, in declaration order
    result: NodeResult                       # the node's AgentResult, including state[DECISION_STATE_KEY]
    history: Sequence[SwarmNode]
    shared_context: Mapping[str, Any]

@dataclass(frozen=True)
class Handoff:
    node: SwarmNode | None                   # None = complete the swarm
    message: str

Swarm(nodes, handoff_strategy=DecisionHandoffStrategy(jev, min_confidence=0.7))
```

`DecisionHandoffStrategy` asks one `Choice` over `candidates` plus `"complete"`. The state is `project_state(result.message, current.system_prompt)` together with each candidate's `name` and `description`. The candidate projection is the same one `DecisionStrategy` uses, so a router and a swarm see identical evidence.

**Precedence.** An explicit `handoff_to_agent` call from the LLM wins, so the strategy runs only when the node ended *without* calling it. This keeps H1's "no change to default behaviour" literally true: with no strategy the swarm behaves as today, and with a strategy an agent that does hand off is never second-guessed.

**Fallback (H2).** A `select` that returns `None` (below the floor, uncalibrated, or a decision error) leaves the state untouched. The swarm then completes, which is exactly today's behaviour when an agent does not hand off. `DecisionHandoffStrategy(fallback="complete" | "stay")` makes that choice explicit. `"stay"` re-runs the current node with an "unsure where to route" note and counts against `max_handoffs`.

**Safety rails unchanged.** A strategy handoff goes through `_handle_handoff`, so `max_handoffs`, `max_iterations`, the repetitive-handoff window, `_turn` rollback and checkpointing all apply as-is. The decision is written to `result.state[DECISION_STATE_KEY]` for the node that ran, so `when_*` helpers and tracing read it from the same place as everywhere else.

**Rejected alternatives.**
- *Replace the handoff tool with the strategy.* This breaks every swarm whose agents rely on the tool's `message`/`context` channel.
- *Let non-`Agent` nodes into Swarm.* That is a larger change: Swarm injects tools into each node's registry. It is unnecessary here, because a `DecisionAgent` entry node can already sit in a `Graph`.
- *Run the strategy before the node.* That is routing, and `ModelRouter` plus `DecisionStrategy` already own it.

### 2. System-One-first fast path

A plugin registers `InvokeModelStage` middleware. This is the same seam `ModelRouter` and the background-tasks plugin use.

```python
FastPath(
    decision_model=jev,
    actions={                                # closed action set: option -> tool call template
        "click_submit": ToolCall("click", {"selector": "#submit"}),
        "scroll_down":  ToolCall("scroll", {"dy": 600}),
    },
    min_confidence=0.8,                      # required; calibrated models only (refuses LLMDecisionModel)
    instructions="Which single next action does `request` need? Choose `other` when none fits.",
)
```

**Per model call.** The middleware asks one `Choice` over `actions ∪ {"other"}` with `project_state(messages, system_prompt)` as state. If the answer is `other`, below `min_confidence`, or an error, it calls `next()` and the LLM runs unchanged. Otherwise it **short-circuits** and yields a synthesized assistant message holding one `toolUse` block built from the template, with stop reason `tool_use`. The loop then executes the tool normally, so tool hooks, interventions (including `DecisionGuard`) and tool spans all run.

**Transcript semantics (H3).** The synthesized message is appended like any assistant turn, because the next LLM call needs the matching `toolUse`/`toolResult` pair. It is marked so that nothing downstream mistakes it for generation:

| Surface               | Marking                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Message               | `message["metadata"]["custom"]["strands"] = {"source": "decision", "decision_model": <id>, "confidence": <c>}`. `MessageMetadata` is stripped before model calls and persisted by session managers, as compression provenance already is. |
| `toolUse.toolUseId`   | `s1-<uuid>` prefix, so a replay or audit can find it without the metadata.                                                                                   |
| Model-invoke span     | No `chat` span. The decision span (`source="decision"`) is the model span for that cycle, with `strands.fast_path.action=<option>`.                          |
| `AfterModelCallEvent` | Fires with `source="decision"` once #4005 adds the field. Until then it does **not** fire, so customer hooks never see an LLM response the LLM did not produce. |
| Usage                 | Decision usage only. No LLM tokens are billed for the cycle.                                                                                                 |

Providers that reject assistant turns they did not produce do not exist in practice (message history is client-owned). Still, the marker lets a conversation manager strip or summarize synthesized turns if a provider starts to care.

**Loop bounds.** The fast path sits inside the normal cycle, so cancellation, interrupts and any loop limits a caller has configured are unchanged. `FastPath(max_consecutive=3)` forces an LLM turn after N consecutive synthesized turns, so a mis-calibrated model cannot drive the agent open-loop indefinitely.

**Why a plugin, not a `Model`.** Wrapping the LLM in a `Model` subclass that sometimes answers from a decision model would make the synthesized turn indistinguishable from model output. That is the attribution problem this design exists to avoid. Middleware keeps the decision on its own span and its own metadata.

### 3. Usage accounting (dependency only)

When #4005 merges, the decision adapters (`DecisionStrategy`, `DecisionAgent`, `DecisionGuard`, `decision_tool`, `DecisionHandoffStrategy`, `FastPath`) call `DecisionModel.ask` through its auxiliary-call runner with `source="decision"`. Then:

- usage rolls into the owning agent's `accumulated_usage`, and
- before- and after-model-call hooks fire, tagged with the source.

`DecisionModel.ask` called directly, outside an adapter, stays span-only, because it has no owning agent. This matches #4005's rule for summarization and routing classifiers.

## Failure modes

| Adapter                   | Uncalibrated model  | Below floor                      | Decision error                   |
| ------------------------- | ------------------- | -------------------------------- | -------------------------------- |
| `DecisionHandoffStrategy` | refused at `__init__` if `min_confidence` is set | `fallback` (default: complete) | `fallback`, warning logged        |
| `FastPath`                | refused at `__init__` (floor required) | pass through to LLM           | pass through to LLM, warning logged |

## Evaluation

Both adapters extend the existing samples rather than adding new ones:

- **Sample 1** (support triage) gains a Swarm arm. It measures handoff accuracy and cost against LLM `handoff_to_agent` on the same labelled tickets.
- **Sample 3** (browser next action) gains a `FastPath` arm. It measures the fraction of steps served by System One, end-to-end task success, and cost, against the all-LLM arm.

Both report the floor's accept and fallback rates, per the #4585 eval conventions.

## Work plan

1. Swarm `handoff_strategy=` and `DecisionHandoffStrategy` (Python, then TS in the same PR). Sample 1 Swarm arm.
2. `FastPath` plugin (Python, then TS), with a transcript-marking test pinned against session round-trip. Sample 3 FastPath arm.
3. After #4005: route adapter decision calls through the auxiliary-call runner (both SDKs).

All three ship under `experimental`, per 0020's lifecycle.
