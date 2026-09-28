import { afterEach, describe, expect, it, vi } from 'vitest'

import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { MockSnapshotStorage } from '../../../__fixtures__/mock-storage-provider.js'
import { Agent } from '../../../agent/agent.js'
import { logger } from '../../../logging/logger.js'
import { AfterNodeCallEvent } from '../../../multiagent/events.js'
import { MultiAgentState, Status } from '../../../multiagent/state.js'
import type { HandoffStrategy } from '../../../multiagent/swarm.js'
import { Swarm } from '../../../multiagent/swarm.js'
import { SessionManager } from '../../../session/session-manager.js'
import type { JSONValue } from '../../../types/json.js'
import { COMPLETE_OPTION, DECISION_STATE_KEY, DecisionHandoffStrategy } from '../decision-handoff-strategy.js'
import { Choice } from '../types.js'
import type { Answer } from '../types.js'
import { MockDecisionModel, choiceAnswer } from './mock-decision-model.js'

/** Answers each `ask` from a queue; an Error entry throws. */
class QueuedDecisionModel extends MockDecisionModel {
  private readonly _queue: Array<Readonly<Record<string, Answer>> | Error>

  constructor(...queue: Array<Readonly<Record<string, Answer>> | Error>) {
    super({})
    this._queue = queue
  }

  protected override async _ask(...args: Parameters<MockDecisionModel['_ask']>): ReturnType<MockDecisionModel['_ask']> {
    const next = this._queue.shift() ?? { next: choiceAnswer(COMPLETE_OPTION, { [COMPLETE_OPTION]: 1 }, 0.9) }
    ;(this as unknown as { _answers: unknown })._answers = next
    return super._ask(...args)
  }
}

function textAgent(id: string, description: string, ...texts: string[]): Agent {
  const model = new MockMessageModel()
  for (const [index, text] of texts.entries()) {
    model.addTurn({
      type: 'toolUseBlock',
      name: 'strands_structured_output',
      toolUseId: `so-${index}`,
      input: { message: text } as JSONValue,
    })
  }
  return new Agent({ model, printer: false, id, description, systemPrompt: `You are ${id}` })
}

function handoffAgent(id: string, description: string, agentId: string): Agent {
  const model = new MockMessageModel().addTurn({
    type: 'toolUseBlock',
    name: 'strands_structured_output',
    toolUseId: 'so-0',
    input: { agentId, message: `over to ${agentId}` } as JSONValue,
  })
  return new Agent({ model, printer: false, id, description, systemPrompt: `You are ${id}` })
}

function team(triage: Agent = textAgent('triage', 'Front line', 'Triaged: a refund request')): Agent[] {
  return [
    triage,
    textAgent('billing', 'Charges and refunds', 'Refund issued'),
    textAgent('technical', 'Bugs and outages', 'Fixed'),
  ]
}

const pick = (key: string, confidence: number): Record<string, Answer> => ({
  next: choiceAnswer(key, { [key]: confidence, other: 1 - confidence }, confidence),
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('DecisionHandoffStrategy', () => {
  it('hands off when the node does not, with bounded state and candidate evidence', async () => {
    const decisions = new QueuedDecisionModel(
      { next: choiceAnswer('c0', { c0: 0.8, c1: 0.1, [COMPLETE_OPTION]: 0.1 }, 0.8) },
      pick(COMPLETE_OPTION, 0.9)
    )
    const swarm = new Swarm({
      nodes: team(),
      handoffStrategy: new DecisionHandoffStrategy(decisions, { minConfidence: 0.7 }),
    })

    const result = await swarm.invoke('I was charged twice')

    expect(result.status).toBe(Status.COMPLETED)
    expect(result.results.map((r) => r.nodeId)).toStrictEqual(['triage', 'billing'])
    const { state, questions } = decisions.requests[0]!
    expect(state).toStrictEqual({
      request: 'I was charged twice',
      agent_instructions: 'You are triage',
      response: 'Triaged: a refund request',
    })
    const options = (questions.next as Choice).options
    expect(Object.keys(options)).toStrictEqual(['c0', 'c1', COMPLETE_OPTION])
    expect([options.c0, options.c1]).toStrictEqual([
      '{"name":"billing","description":"Charges and refunds"}',
      '{"name":"technical","description":"Bugs and outages"}',
    ])
    expect(result.results[0]!.structuredOutput).toStrictEqual({
      agentId: 'billing',
      message:
        'triage finished without handing off; a decision model routed the task to you (confidence 0.80).' +
        "\n\ntriage's final response:\nTriaged: a refund request",
    })
  })

  it('records each node decision on state.app keyed by node id', async () => {
    const decisions = new QueuedDecisionModel(
      { next: choiceAnswer('c1', { c0: 0.2, c1: 0.7, [COMPLETE_OPTION]: 0.1 }, 0.7) },
      pick(COMPLETE_OPTION, 0.8)
    )
    const swarm = new Swarm({ nodes: team(), handoffStrategy: new DecisionHandoffStrategy(decisions) })
    let captured: MultiAgentState | undefined
    swarm.addHook(AfterNodeCallEvent, (event) => {
      captured = event.state
    })

    await swarm.invoke('The export button 500s')

    const recorded = captured!.app.get(DECISION_STATE_KEY) as Record<string, Record<string, JSONValue>>
    expect(Object.keys(recorded)).toStrictEqual(['triage', 'technical'])
    expect(recorded.triage!.output).toStrictEqual({ next: 'technical' })
    expect(recorded.triage!.answers).toMatchObject({
      next: { choice: 'technical', probabilities: { billing: 0.2, technical: 0.7, complete: 0.1 }, confidence: 0.7 },
    })
    expect(recorded.technical!.output).toStrictEqual({ next: COMPLETE_OPTION })
    const triageResults = captured!.node('triage')!.results
    expect(triageResults[triageResults.length - 1]).toBe(captured!.results[0])
    expect((captured!.results[0]!.structuredOutput as { agentId: string }).agentId).toBe('technical')
  })

  it('never second-guesses a handoff the agent chose itself', async () => {
    const decisions = new QueuedDecisionModel(pick(COMPLETE_OPTION, 0.9))
    const swarm = new Swarm({
      nodes: team(handoffAgent('triage', 'Front line', 'technical')),
      handoffStrategy: new DecisionHandoffStrategy(decisions, { minConfidence: 0.5 }),
    })

    const result = await swarm.invoke('The export button 500s')

    expect(result.results.map((r) => r.nodeId)).toStrictEqual(['triage', 'technical'])
    expect(decisions.requests.map((r) => (r.state as Record<string, string>).agent_instructions)).toStrictEqual([
      'You are technical',
    ])
  })

  it('completes on a complete answer', async () => {
    const decisions = new QueuedDecisionModel(pick(COMPLETE_OPTION, 0.95))
    const swarm = new Swarm({ nodes: team(), handoffStrategy: new DecisionHandoffStrategy(decisions) })

    const result = await swarm.invoke('Thanks, all sorted')

    expect(result.results.map((r) => r.nodeId)).toStrictEqual(['triage'])
  })

  it.each([
    ['complete', ['triage']],
    ['stay', ['triage', 'triage']],
  ] as const)('below the floor uses fallback %s', async (fallback, expected) => {
    const decisions = new QueuedDecisionModel(
      { next: choiceAnswer('c0', { c0: 0.4, c1: 0.35, [COMPLETE_OPTION]: 0.25 }, 0.4) },
      pick(COMPLETE_OPTION, 0.9)
    )
    const swarm = new Swarm({
      nodes: team(textAgent('triage', 'Front line', 'Not sure yet', 'Resolved it myself')),
      handoffStrategy: new DecisionHandoffStrategy(decisions, { minConfidence: 0.7, fallback }),
    })

    const result = await swarm.invoke('Something is off')

    expect(result.status).toBe(Status.COMPLETED)
    expect(result.results.map((r) => r.nodeId)).toStrictEqual(expected)
    if (fallback === 'stay') {
      expect((result.results[0]!.structuredOutput as { message: string }).message).toContain(
        'could not choose the next agent with enough confidence'
      )
    }
  })

  it.each([
    ['complete', ['triage']],
    ['stay', ['triage', 'triage']],
  ] as const)('a decision error uses fallback %s and warns', async (fallback, expected) => {
    const warn = vi.spyOn(logger, 'warn')
    const decisions = new QueuedDecisionModel(new Error('service down'), pick(COMPLETE_OPTION, 0.9))
    const swarm = new Swarm({
      nodes: team(textAgent('triage', 'Front line', 'First pass', 'Second pass')),
      handoffStrategy: new DecisionHandoffStrategy(decisions, { minConfidence: 0.7, fallback }),
    })

    const result = await swarm.invoke('Something is off')

    expect(result.status).toBe(Status.COMPLETED)
    expect(result.results.map((r) => r.nodeId)).toStrictEqual(expected)
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('error_type=<Error> | handoff decision failed'))
  })

  it('bounds a stay fallback by maxSteps', async () => {
    const decisions = new QueuedDecisionModel(...Array.from({ length: 5 }, () => pick('c0', 0.1)))
    const swarm = new Swarm({
      nodes: team(textAgent('triage', 'Front line', 'p0', 'p1', 'p2', 'p3', 'p4')),
      maxSteps: 3,
      handoffStrategy: new DecisionHandoffStrategy(decisions, { minConfidence: 0.7, fallback: 'stay' }),
    })

    await expect(swarm.invoke('Loop forever')).rejects.toThrow('max_steps=<3> | swarm reached step limit')
  })

  it('trips repetitive-handoff detection on strategy handoffs', async () => {
    const decisions = new QueuedDecisionModel(...Array.from({ length: 4 }, () => pick('c0', 0.9)))
    const swarm = new Swarm({
      nodes: [
        textAgent('triage', 'Front line', 't1', 't2'),
        textAgent('billing', 'Charges', 'b1', 'b2'),
        textAgent('technical', 'Bugs', 'x'),
      ],
      repetitiveHandoffDetectionWindow: 4,
      repetitiveHandoffMinUniqueAgents: 3,
      handoffStrategy: new DecisionHandoffStrategy(decisions),
    })

    const result = await swarm.invoke('Ping pong')

    expect(result.status).toBe(Status.FAILED)
    expect(result.results.map((r) => r.nodeId)).toStrictEqual(['triage', 'billing', 'triage', 'billing'])
  })

  it('resumes a saved session at the strategy handoff target', async () => {
    const storage = new MockSnapshotStorage()
    const decisions = new QueuedDecisionModel(pick('c0', 0.9))
    const swarm = new Swarm({
      nodes: [textAgent('triage', 'Front line', 'triaged'), textAgent('billing', 'Charges', 'unused')],
      sessionManager: new SessionManager({ sessionId: 's1', storage: { snapshot: storage } }),
      handoffStrategy: new DecisionHandoffStrategy(decisions),
    })
    swarm.addHook(AfterNodeCallEvent, (event) => {
      if (event.nodeId === 'triage') throw new Error('crash after triage')
    })
    await expect(swarm.invoke('I was charged twice')).rejects.toThrow('crash after triage')

    const resumed = new Swarm({
      nodes: [textAgent('triage', 'Front line', 'unused'), textAgent('billing', 'Charges', 'refunded after restart')],
      sessionManager: new SessionManager({ sessionId: 's1', storage: { snapshot: storage } }),
      handoffStrategy: new DecisionHandoffStrategy(new QueuedDecisionModel(pick(COMPLETE_OPTION, 0.9))),
    })
    const result = await resumed.invoke('I was charged twice')

    expect(result.results.map((r) => r.nodeId)).toStrictEqual(['triage', 'billing'])
    expect(result.content.map((block) => (block as { text: string }).text)).toStrictEqual(['refunded after restart'])
  })

  it('refuses a node from another swarm', async () => {
    const other = new Swarm({ nodes: team() })
    const strategy: HandoffStrategy = { select: async () => ({ node: other.nodes.get('billing')!, message: 'go' }) }
    const swarm = new Swarm({ nodes: team(), handoffStrategy: strategy })

    await expect(swarm.invoke('anything')).rejects.toThrow('handoff strategy chose a node that is not in this swarm')
  })

  it('refuses a node id that collides with the complete option', async () => {
    const swarm = new Swarm({
      nodes: [textAgent('triage', 'Front line', 'done'), textAgent(COMPLETE_OPTION, 'x', 'x')],
      handoffStrategy: new DecisionHandoffStrategy(new QueuedDecisionModel()),
    })

    await expect(swarm.invoke('anything')).rejects.toThrow("collides with the handoff decision's complete option")
  })

  it('validates its options', () => {
    expect(() => new DecisionHandoffStrategy(new MockDecisionModel({}, false), { minConfidence: 0.7 })).toThrow(
      'DecisionHandoffStrategy(minConfidence=0.7) needs a calibrated DecisionModel'
    )
    expect(
      () => new DecisionHandoffStrategy(new MockDecisionModel(), { fallback: 'guess' as unknown as 'stay' })
    ).toThrow("fallback must be 'complete' or 'stay'")
    expect(() => new DecisionHandoffStrategy(new MockDecisionModel(), { maxResponseTokens: 0 })).toThrow(
      'maxResponseTokens must be a positive integer'
    )
  })

  it('leaves a swarm without a strategy unchanged', async () => {
    const result = await new Swarm({ nodes: team() }).invoke('anything')

    expect(result.results.map((r) => r.nodeId)).toStrictEqual(['triage'])
  })
})
