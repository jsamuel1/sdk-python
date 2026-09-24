import { describe, expect, it } from 'vitest'
import { z } from 'zod'

import { DecisionModel } from '../decision-model.js'
import type { DecisionModelConfig } from '../decision-model.js'
import { choice, yesNo } from '../schema.js'
import { Choice, ChoiceAnswer, DecisionResponse } from '../types.js'
import type { Answer, DecisionState, Question } from '../types.js'
import { MockDecisionModel, choiceAnswer, yes } from './mock-decision-model.js'

const Triage = z.object({
  department: choice(['billing', 'technical'], 'Which team?'),
  urgent: yesNo('Urgent?'),
})

class BadProvider extends MockDecisionModel {
  constructor(private readonly _reply: Readonly<Record<string, Answer>>) {
    super()
  }

  protected override async _ask(
    _state: DecisionState,
    _questions: Readonly<Record<string, Question>>
  ): Promise<DecisionResponse> {
    return new DecisionResponse(this._reply)
  }
}

describe('DecisionModel', () => {
  describe('decide', () => {
    it('asks every field in one request and returns a typed decision', async () => {
      const answers = { department: choiceAnswer('billing', { billing: 0.8, technical: 0.2 }, 0.6), urgent: yes(0.9) }
      const model = new MockDecisionModel(answers)

      const decision = await model.decide(Triage, { ticket: 'charged twice' })

      expect(decision).toStrictEqual({
        output: { department: 'billing', urgent: true },
        answers,
        modelId: 'mock-s1-1.0',
        usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12 },
      })
      expect(model.requests).toHaveLength(1)
      expect(Object.keys(model.requests[0]!.questions)).toStrictEqual(['department', 'urgent'])
    })
  })

  describe('ask', () => {
    it('rejects empty questions and an option-less Choice', async () => {
      const model = new MockDecisionModel()

      await expect(model.ask('s', {})).rejects.toThrow('at least one question')
      await expect(model.ask('s', { q: new Choice('pick') })).rejects.toThrow('have no options')
    })

    it.each([
      ['a missing answer', {}, 'no answer for [q]'],
      ['the wrong answer type', { q: yes(1) }, 'expected ChoiceAnswer'],
      ['an unknown option', { q: new ChoiceAnswer('zzz', { zzz: 1 }) }, 'not one of the options'],
    ])('rejects %s from the provider', async (_name, reply, message) => {
      await expect(new BadProvider(reply).ask('s', { q: new Choice('pick', { a: null }) })).rejects.toThrow(message)
    })

    it('propagates provider errors', async () => {
      const model = new MockDecisionModel(new Error('boom'))

      await expect(model.ask('s', { q: new Choice('pick', { a: null }) })).rejects.toThrow('boom')
    })
  })

  it('defaults to uncalibrated and reads modelId from config', () => {
    class Plain extends DecisionModel {
      getConfig(): DecisionModelConfig {
        return { modelId: 'plain-1' }
      }
      updateConfig(): void {}
      protected async _ask(): Promise<DecisionResponse> {
        return new DecisionResponse({})
      }
    }

    const model = new Plain()

    expect({ calibrated: model.calibrated, modelId: model.modelId }).toStrictEqual({
      calibrated: false,
      modelId: 'plain-1',
    })
  })
})
