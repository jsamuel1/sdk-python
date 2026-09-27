import { describe, expect, it } from 'vitest'

import { DecisionModel } from '../decision-model.js'
import { answerFromLogits, checkTemperature, logitsOf, softmax } from '../calibration.js'
import { Choice, ChoiceAnswer, Score, ScoreAnswer, YesNo, YesNoAnswer, maxProbabilityConfidence } from '../types.js'

describe('answerFromLogits', () => {
  it('builds a Choice answer at a temperature and keeps the logits and extras', () => {
    const logits = { billing: 2, technical: 0 }

    const answer = answerFromLogits(new Choice('pick', { billing: null, technical: null }), logits, {
      temperature: 2,
      extras: { latencyMs: 12 },
    }) as ChoiceAnswer

    const billing = Math.exp(1) / (Math.exp(1) + 1)
    expect(answer.choice).toBe('billing')
    expect(answer.probabilities.billing).toBeCloseTo(billing, 12)
    expect(answer.confidence).toBeCloseTo(billing, 12)
    expect({ logits: answer.logits, extras: answer.extras }).toStrictEqual({ logits, extras: { latencyMs: 12 } })
  })

  it('builds a Score answer whose score is the probability-weighted level', () => {
    const answer = answerFromLogits(new Score('rate', ['a', 'b', 'c']), { 0: 0, 1: 0, 2: 0 }) as ScoreAnswer

    expect(answer.score).toBeCloseTo(1, 12)
    expect(answer.confidence).toBeCloseTo(1 / 3, 12)
  })

  it('builds a YesNo answer from true/false logits', () => {
    const answer = answerFromLogits(new YesNo('ok?'), { true: 0, false: 0 }) as YesNoAnswer

    expect({ probability: answer.probability, confidence: answer.confidence }).toStrictEqual({
      probability: 0.5,
      confidence: 0,
    })
  })

  it('reports no confidence when uncalibrated', () => {
    const answer = answerFromLogits(new Choice('pick', { a: null, b: null }), { a: 1, b: 0 }, { calibrated: false })

    expect(answer.confidence).toBeUndefined()
  })

  it.each([
    ['a zero temperature', new Choice('p', { a: null }), { a: 1 }, { temperature: 0 }, 'finite number above 0'],
    ['empty logits', new Choice('p', { a: null }), {}, {}, 'at least one logit'],
    ['YesNo logits without true', new YesNo('q'), { yes: 1, no: 0 }, {}, "keyed 'true' and 'false'"],
  ])('rejects %s', (_name, question, logits, options, message) => {
    expect(() => answerFromLogits(question, logits, options)).toThrow(message)
  })

  it('is exposed on the DecisionModel base', () => {
    expect(DecisionModel.answerFromLogits).toBe(answerFromLogits)
  })
})

describe('temperature helpers', () => {
  it('softmax over logitsOf recovers the distribution, and T rescales it as p^(1/T)', () => {
    const logits = logitsOf({ a: 0.9, b: 0.1 })

    expect(softmax(logits).a).toBeCloseTo(0.9, 12)
    expect(softmax(logits, 2).a).toBeCloseTo(Math.sqrt(0.9) / (Math.sqrt(0.9) + Math.sqrt(0.1)), 12)
    expect(Number.isFinite(logitsOf({ zero: 0 }).zero)).toBe(true)
  })

  it.each([0, -2, Number.NaN, Number.POSITIVE_INFINITY, '1'])('checkTemperature rejects %s', (value) => {
    expect(() => checkTemperature(value)).toThrow('temperature must be a finite number above 0')
  })
})

describe('answer raw scores', () => {
  it('defaults extras to {} and leaves logits undefined', () => {
    const answer = new ChoiceAnswer('a', { a: 1 })

    expect({ logits: answer.logits, extras: answer.extras }).toStrictEqual({ logits: undefined, extras: {} })
  })

  it.each([
    ['mismatched Choice keys', (): unknown => new ChoiceAnswer('a', { a: 1 }, 1, { logits: { b: 1 } })],
    ['non-finite Score logits', (): unknown => new ScoreAnswer(0, { 0: 1 }, 1, { logits: { 0: Number.NaN } })],
    [
      'YesNo logits not keyed true/false',
      (): unknown => new YesNoAnswer(0.5, 0, { logits: { yes: 0 } as unknown as { true: number; false: number } }),
    ],
  ])('rejects %s', (_name, build) => {
    expect(build).toThrow('answer logits must be finite and keyed exactly like the probabilities')
  })

  it('maxProbabilityConfidence is the top probability and rejects an empty map', () => {
    expect(maxProbabilityConfidence({ a: 0.2, b: 0.7, c: 0.1 })).toBe(0.7)
    expect(() => maxProbabilityConfidence({})).toThrow('at least one probability')
  })
})
