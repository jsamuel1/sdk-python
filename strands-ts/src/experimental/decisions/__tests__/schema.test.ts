import { describe, expect, it } from 'vitest'
import { z } from 'zod'

import { NO_MATCH_OPTION, choice, compileSchema, score, yesNo } from '../schema.js'
import { Choice, ChoiceAnswer, Score, ScoreAnswer, YesNo, YesNoAnswer, yesNoConfidence } from '../types.js'
import { choiceAnswer, scoreAnswer, yes } from './mock-decision-model.js'

const Triage = z.object({
  department: choice({ billing: 'Payments', technical: null }).describe('Which team owns `ticket`?'),
  urgent: yesNo('Is it urgent?', { threshold: 0.7 }),
  severity: score(['minor', 'major', 'critical'], 'How bad is it?'),
})

describe('compileSchema', () => {
  it('builds questions from field types and markers', () => {
    const { questions } = compileSchema(Triage)

    expect(questions).toStrictEqual({
      department: new Choice('Which team owns `ticket`?', { billing: 'Payments', technical: null }),
      urgent: new YesNo('Is it urgent?', { threshold: 0.7 }),
      severity: new Score('How bad is it?', ['minor', 'major', 'critical']),
    })
  })

  it('caches the compiled schema per schema object', () => {
    expect(compileSchema(Triage)).toBe(compileSchema(Triage))
  })

  it('compiles plain z.enum and z.boolean fields from .describe()', () => {
    const schema = z.object({ tier: z.enum(['a', 'b']).describe('Pick a tier'), ok: z.boolean().describe('OK?') })

    expect(compileSchema(schema).questions).toStrictEqual({
      tier: new Choice('Pick a tier', { a: null, b: null }),
      ok: new YesNo('OK?'),
    })
  })

  it('adds the no-match option to an optional choice', () => {
    const schema = z.object({ tier: z.enum(['a', 'b']).optional().describe('Pick') })

    expect(compileSchema(schema).questions.tier).toStrictEqual(
      new Choice('Pick', { a: null, b: null, [NO_MATCH_OPTION]: 'None of the other options applies' })
    )
  })

  it.each([
    ['a string field', z.object({ name: z.string().describe('x') }), "unsupported field type 'string'"],
    ['a plain number', z.object({ n: z.number().describe('x') }), 'needs score'],
    ['an optional boolean', z.object({ b: z.boolean().optional().describe('x') }), 'cannot be optional'],
    ['an optional score', z.object({ s: score(['a', 'b'], 'x').optional() }), 'cannot be optional'],
    ['no instructions', z.object({ b: z.boolean() }), 'give the field instructions'],
    ['a reserved option', z.object({ c: z.enum(['a', 'none']).optional().describe('x') }), 'reserves'],
    ['an empty schema', z.object({}), 'no fields'],
  ])('rejects %s', (_name, schema, message) => {
    expect(() => compileSchema(schema)).toThrow(message)
  })

  it('rejects a non-object schema', () => {
    expect(() => compileSchema(z.string() as unknown as z.ZodObject)).toThrow(TypeError)
  })
})

describe('CompiledSchema.buildOutput', () => {
  it('maps answers to typed fields', () => {
    const output = compileSchema(Triage).buildOutput({
      department: choiceAnswer('billing', { billing: 0.8, technical: 0.2 }, 0.6),
      urgent: yes(0.65),
      severity: scoreAnswer(1.4, { 0: 0.1, 1: 0.4, 2: 0.5 }),
    })

    expect(output).toStrictEqual({ department: 'billing', urgent: false, severity: 1.4 })
  })

  it.each([
    ['optional', z.enum(['a']).optional(), undefined],
    ['nullable', z.enum(['a']).nullable(), null],
  ])('maps the no-match option on a %s choice', (_name, field, expected) => {
    const schema = z.object({ tier: field.describe('Pick') })

    const output = compileSchema(schema).buildOutput({ tier: choiceAnswer(NO_MATCH_OPTION, { a: 0.1, none: 0.9 }) })

    expect(output.tier).toBe(expected)
  })

  it('rejects a missing or mistyped answer', () => {
    const compiled = compileSchema(Triage)

    expect(() => compiled.buildOutput({})).toThrow('no answer for department')
    expect(() =>
      compiled.buildOutput({ department: yes(0.9), urgent: yes(0.9), severity: scoreAnswer(1, { 1: 1 }) })
    ).toThrow('expected a choice answer')
  })
})

describe('question and answer types', () => {
  it.each([
    ['empty Choice instructions', (): unknown => new Choice(''), 'instructions'],
    ['a one-level Score', (): unknown => new Score('rate', ['only']), 'at least 2 levels'],
    ['a YesNo threshold above 1', (): unknown => new YesNo('q', { threshold: 1.5 }), 'between 0 and 1'],
    ['a choice outside its distribution', (): unknown => new ChoiceAnswer('z', { a: 1 }), 'not one of'],
    ['a negative probability', (): unknown => new ScoreAnswer(1, { 0: -1 }), 'non-negative'],
    ['a YesNo probability above 1', (): unknown => new YesNoAnswer(1.2), 'between 0 and 1'],
    ['an empty choice() marker', (): unknown => choice([] as unknown as ['a']), 'at least one option'],
  ])('rejects %s', (_name, build, message) => {
    expect(build).toThrow(message)
  })

  it('carries no vendor limits on question shape', () => {
    const options = Object.fromEntries(Array.from({ length: 300 }, (_value, index) => [`o${index}`, null]))

    expect(Object.keys(new Choice('pick', options).options)).toHaveLength(300)
    expect(
      new Score(
        'rate',
        Array.from({ length: 20 }, (_value, index) => `l${index}`)
      ).levels
    ).toHaveLength(20)
  })

  it('returns the most probable level', () => {
    expect(new ScoreAnswer(1.2, { 0: 0.2, 1: 0.5, 2: 0.3 }).level).toBe(1)
  })

  it.each([
    [0.5, 0],
    [1, 1],
    [0, 1],
    [0.75, 0.5],
  ])('yesNoConfidence(%s) is %s', (probability, expected) => {
    expect(yesNoConfidence(probability)).toBe(expected)
  })
})
