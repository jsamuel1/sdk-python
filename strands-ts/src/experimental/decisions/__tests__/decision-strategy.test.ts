import { afterEach, describe, expect, it, vi } from 'vitest'

import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { Agent } from '../../../agent/agent.js'
import { logger } from '../../../logging/logger.js'
import { ModelRouter, RoutingCandidate } from '../../../models/routing/router.js'
import type { RoutingContext } from '../../../models/routing/strategy.js'
import { Message, TextBlock } from '../../../types/messages.js'
import { DecisionStrategy } from '../decision-strategy.js'
import { Choice } from '../types.js'
import { MockDecisionModel, choiceAnswer } from './mock-decision-model.js'

function responseModel(text: string): MockMessageModel {
  return new MockMessageModel().addTurn({ type: 'textBlock', text })
}

function router(strategy: DecisionStrategy): ModelRouter {
  return new ModelRouter(
    [
      new RoutingCandidate({ model: responseModel('routine'), name: 'routine', description: 'Simple lookups' }),
      new RoutingCandidate({
        model: responseModel('complex'),
        name: 'complex',
        description: 'Multi-step reasoning',
        metadata: { tier: 3 },
      }),
    ],
    { strategy }
  )
}

function context(target: ModelRouter, attempts: RoutingContext['attempts'] = []): RoutingContext {
  return {
    messages: [new Message({ role: 'user', content: [new TextBlock('Prove sqrt(2) is irrational')] })],
    systemPrompt: 'Be precise',
    toolSpecs: [],
    candidates: target.candidates,
    invocationState: {},
    attempts,
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('DecisionStrategy', () => {
  describe('select', () => {
    it('routes the agent to the chosen candidate with bounded state and candidate evidence', async () => {
      const decisions = new MockDecisionModel({ candidate: choiceAnswer('c1', { c0: 0.1, c1: 0.9 }, 0.85) })
      const agent = new Agent({ model: router(new DecisionStrategy(decisions)), printer: false })

      const result = await agent.invoke('Prove sqrt(2) is irrational')

      expect(result.toString().trim()).toBe('complex')
      const { state, questions } = decisions.requests[0]!
      expect(state).toStrictEqual({ request: 'Prove sqrt(2) is irrational', agent_instructions: '' })
      expect((questions.candidate as Choice).options).toStrictEqual({
        c0: '{"name":"routine","description":"Simple lookups"}',
        c1: '{"name":"complex","description":"Multi-step reasoning","metadata":{"tier":3}}',
      })
    })

    it('sends the system prompt as agent instructions', async () => {
      const decisions = new MockDecisionModel({ candidate: choiceAnswer('c0', { c0: 0.9, c1: 0.1 }, 0.9) })
      const strategy = new DecisionStrategy(decisions)
      const target = router(strategy)

      expect(await strategy.select(context(target))).toBe(target.candidates[0])
      expect(decisions.requests[0]!.state).toStrictEqual({
        request: 'Prove sqrt(2) is irrational',
        agent_instructions: 'Be precise',
      })
    })

    it('declines on low confidence', async () => {
      const decisions = new MockDecisionModel({ candidate: choiceAnswer('c1', { c0: 0.45, c1: 0.55 }, 0.1) })
      const strategy = new DecisionStrategy(decisions, { minConfidence: 0.7 })

      expect(await strategy.select(context(router(strategy)))).toBeUndefined()
    })

    it('declines and warns on a decision error, and declines after a failure', async () => {
      const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
      const strategy = new DecisionStrategy(new MockDecisionModel(new Error('boom')))
      const target = router(strategy)

      expect(await strategy.select(context(target))).toBeUndefined()
      expect(warn).toHaveBeenCalledWith(expect.stringContaining('reason=<decision_error>'))
      const failed = [{ candidate: target.candidates[0]!, exception: new Error('x') }]
      expect(await strategy.select(context(target, failed))).toBeUndefined()
    })

    it('skips the decision for a single candidate', async () => {
      const decisions = new MockDecisionModel()
      const strategy = new DecisionStrategy(decisions)
      const target = new ModelRouter([responseModel('only')], { strategy })

      expect(await strategy.select(context(target))).toBe(target.candidates[0])
      expect(decisions.requests).toStrictEqual([])
    })
  })

  describe('constructor', () => {
    it('rejects minConfidence on an uncalibrated model', () => {
      expect(() => new DecisionStrategy(new MockDecisionModel({}, false), { minConfidence: 0.7 })).toThrow(
        'DecisionStrategy(minConfidence=0.7) needs a calibrated DecisionModel'
      )
    })

    it.each([
      ['maxRequestTokens', { maxRequestTokens: 0 }],
      ['maxInstructionTokens', { maxInstructionTokens: 1.5 }],
    ])('rejects a non-positive-integer %s', (name, options) => {
      expect(() => new DecisionStrategy(new MockDecisionModel(), options)).toThrow(`${name} must be a positive integer`)
    })

    it('rejects a non-DecisionModel', () => {
      expect(() => new DecisionStrategy({} as MockDecisionModel)).toThrow(TypeError)
    })
  })
})
