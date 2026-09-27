import { afterEach, describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { TypeSafeClient } from '@typesafe-ai/sdk'
import type { Fetch } from '@typesafe-ai/sdk'

import { ContextWindowOverflowError, ModelThrottledError } from '../../errors.js'
import { DecisionStrategy } from '../../experimental/decisions/decision-strategy.js'
import { choice, score, yesNo } from '../../experimental/decisions/schema.js'
import { Choice, ChoiceAnswer, Score, ScoreAnswer, YesNo, YesNoAnswer } from '../../experimental/decisions/types.js'
import {
  DEFAULT_BASE_URL,
  KEV_MAX_STATE_PLUS_QUESTION_TOKENS,
  MAX_CHOICE_OPTIONS,
  MAX_SCORE_LEVELS,
  TypeSafeDecisionModel,
} from '../typesafe.js'

interface Captured {
  url: string
  headers: Record<string, string>
  body: Record<string, unknown>
}

const OK_BODY = {
  model: 'jev-1.13.0',
  answers: {
    dept: { type: 'choice', choice: 'billing', confidence: 0.9, probabilities: { billing: 0.95, technical: 0.05 } },
    urgent: { type: 'noul', noul: 0.8 },
    severity: { type: 'score', score: 1.2, confidence: 0.4, legend: {}, probabilities: { 0: 0.1, 1: 0.6, 2: 0.3 } },
  },
  usage: { input_tokens: 40, output_tokens: 3 },
}

function fakeFetch(status: number, body: unknown, captured: Captured[] = []): Fetch {
  return vi.fn(async (input: string, init?: RequestInit) => {
    captured.push({
      url: String(input),
      headers: Object.fromEntries(new Headers(init?.headers).entries()),
      body: JSON.parse(String(init?.body)) as Record<string, unknown>,
    })
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
  }) as unknown as Fetch
}

function model(status: number, body: unknown, captured: Captured[] = []): TypeSafeDecisionModel {
  return new TypeSafeDecisionModel({
    apiKey: 'k',
    modelId: 'jev-1.13.0',
    clientConfig: { fetch: fakeFetch(status, body, captured), retry: { maxRetries: 0 }, logLevel: 'off' },
  })
}

const QUESTIONS = {
  dept: new Choice('Which team?', { billing: 'Payments', technical: null }),
  urgent: new YesNo('Urgent?', { true: 'needs action today' }),
  severity: new Score('How bad?', ['minor', 'major', 'critical']),
}

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('TypeSafeDecisionModel', () => {
  describe('ask', () => {
    it('maps questions to the wire and answers back to typed answers', async () => {
      const captured: Captured[] = []

      const response = await model(200, OK_BODY, captured).ask({ ticket: 'charged twice' }, QUESTIONS)

      expect(captured[0]!.url).toBe(`${DEFAULT_BASE_URL}/v1/systemone`)
      expect(captured[0]!.body).toStrictEqual({
        state: { ticket: 'charged twice' },
        model: 'jev-1.13.0',
        questions: {
          dept: { type: 'choice', instructions: 'Which team?', criteria: { billing: 'Payments', technical: null } },
          urgent: { type: 'noul', instructions: 'Urgent?', criteria: { true: 'needs action today' } },
          severity: { type: 'score', instructions: 'How bad?', criteria: ['minor', 'major', 'critical'] },
        },
      })
      expect(response.answers).toStrictEqual({
        dept: new ChoiceAnswer('billing', { billing: 0.95, technical: 0.05 }, 0.9),
        urgent: new YesNoAnswer(0.8, 0.6000000000000001),
        severity: new ScoreAnswer(1.2, { 0: 0.1, 1: 0.6, 2: 0.3 }, 0.4),
      })
      expect({ modelId: response.modelId, usage: response.usage }).toStrictEqual({
        modelId: 'jev-1.13.0',
        usage: { inputTokens: 40, outputTokens: 3, totalTokens: 43 },
      })
    })

    it('sends a YesNo without criteria when it has no outcome descriptions', async () => {
      const captured: Captured[] = []
      const body = { ...OK_BODY, answers: { q: { type: 'noul', noul: 0.2 } } }

      await model(200, body, captured).ask('s', { q: new YesNo('ok?') })

      expect((captured[0]!.body.questions as Record<string, unknown>).q).toStrictEqual({
        type: 'noul',
        instructions: 'ok?',
      })
    })

    it.each([
      [429, { error: 'slow down' }, ModelThrottledError],
      [529, { error: 'overloaded' }, ModelThrottledError],
      [400, { error: 'bad' }, Error, 'TypeSafe rejected the decision request'],
      [422, { error: 'invalid' }, Error, 'TypeSafe rejected the decision request'],
    ])('maps HTTP %s to a typed error with the vendor cause', async (status, body, expected, message?: string) => {
      const error = (await model(status, body)
        .ask('s', QUESTIONS)
        .then(
          () => undefined,
          (caught: unknown) => caught
        )) as Error

      expect(error).toBeInstanceOf(expected)
      if (message) expect(error.message).toContain(message)
      expect(error.cause).toBeDefined()
    })

    it('lets other API errors through unchanged', async () => {
      await expect(model(401, { error: 'no' }).ask('s', QUESTIONS)).rejects.toThrow(/401|auth/i)
    })

    it.each([
      ['a state over the row budget', 'x'.repeat(200_000), QUESTIONS, 'longest question'],
      [
        'a request over the total budget',
        'x'.repeat(100_000),
        Object.fromEntries(Array.from({ length: 60 }, (_value, index) => [`q${index}`, new YesNo('y'.repeat(4_000))])),
        'per request',
      ],
    ])('rejects %s before sending', async (_name, state, questions, message) => {
      const captured: Captured[] = []

      await expect(model(200, OK_BODY, captured).ask(state, questions)).rejects.toThrow(ContextWindowOverflowError)
      await expect(model(200, OK_BODY, captured).ask(state, questions)).rejects.toThrow(message)
      expect(captured).toStrictEqual([])
    })

    it.each([
      [
        'Choice options',
        new Choice(
          'pick',
          Object.fromEntries(Array.from({ length: MAX_CHOICE_OPTIONS + 1 }, (_v, i) => [`o${i}`, null]))
        ),
      ],
      [
        'Score levels',
        new Score(
          'rate',
          Array.from({ length: MAX_SCORE_LEVELS + 1 }, (_v, i) => `l${i}`)
        ),
      ],
    ])('enforces the TypeSafe limit on %s before sending', async (name, question) => {
      const captured: Captured[] = []

      await expect(model(200, OK_BODY, captured).ask('s', { q: question })).rejects.toThrow('at most')
      await expect(model(200, OK_BODY, captured).ask('s', { q: question })).rejects.toThrow(name)
      expect(captured).toStrictEqual([])
    })

    it('rejects an unknown vendor answer type', async () => {
      const body = { ...OK_BODY, answers: { q: { type: 'mystery' } } }

      await expect(model(200, body).ask('s', { q: new YesNo('ok?') })).rejects.toThrow(
        "unsupported TypeSafe answer type 'mystery'"
      )
    })
  })

  it('decides a Zod schema end to end with a versioned model id', async () => {
    const Triage = z.object({
      dept: choice({ billing: 'Payments', technical: null }, 'Which team?'),
      urgent: yesNo('Urgent?'),
      severity: score(['minor', 'major', 'critical'], 'How bad?'),
    })

    const decision = await model(200, OK_BODY).decide(Triage, 'charged twice')

    expect({ output: decision.output, modelId: decision.modelId }).toStrictEqual({
      output: { dept: 'billing', urgent: true, severity: 1.2 },
      modelId: 'jev-1.13.0',
    })
  })

  describe('constructor', () => {
    it('requires an API key for the TypeSafe server', () => {
      vi.stubEnv('TYPESAFE_API_KEY', '')
      vi.stubEnv('TYPESAFE_BASE_URL', '')

      expect(() => new TypeSafeDecisionModel()).toThrow('needs an API key')
    })

    it('reads the key from TYPESAFE_API_KEY and the server from TYPESAFE_BASE_URL', async () => {
      vi.stubEnv('TYPESAFE_API_KEY', 'env-key')
      const captured: Captured[] = []
      const fetch = fakeFetch(200, { ...OK_BODY, answers: { q: { type: 'noul', noul: 0.5 } } }, captured)

      await new TypeSafeDecisionModel({ clientConfig: { fetch, logLevel: 'off' } }).ask('s', { q: new YesNo('ok?') })

      expect(captured[0]!.headers.authorization).toBe('Bearer env-key')
    })

    it('never sends TYPESAFE_API_KEY to a self-hosted server such as Kev', async () => {
      vi.stubEnv('TYPESAFE_API_KEY', 'typesafe-secret')
      vi.stubEnv('TYPESAFE_BASE_URL', 'http://127.0.0.1:8009/')
      const captured: Captured[] = []
      const fetch = fakeFetch(200, { ...OK_BODY, answers: { q: { type: 'noul', noul: 0.5 } } }, captured)

      const kev = new TypeSafeDecisionModel({
        modelId: 'kev-latest',
        maxStatePlusQuestionTokens: KEV_MAX_STATE_PLUS_QUESTION_TOKENS,
        maxRequestTokens: null,
        clientConfig: { fetch, logLevel: 'off' },
      })
      await kev.ask('s', { q: new YesNo('ok?') })

      expect({ baseUrl: kev.baseUrl, url: captured[0]!.url, auth: captured[0]!.headers.authorization }).toStrictEqual({
        baseUrl: 'http://127.0.0.1:8009',
        url: 'http://127.0.0.1:8009/v1/systemone',
        auth: 'Bearer local',
      })
    })

    it('uses a caller-supplied client and forwards the cancel signal', async () => {
      const client = new TypeSafeClient({ apiKey: 'k', logLevel: 'off' })
      const systemOne = vi.spyOn(client, 'systemOne').mockReturnValue(
        Promise.resolve({
          model: 'jev',
          answers: { q: { type: 'noul', noul: 1 } },
          usage: { input_tokens: 1, output_tokens: 0 },
        }) as never
      )
      const signal = new AbortController().signal

      await new TypeSafeDecisionModel({ client }).ask('s', { q: new YesNo('ok?') }, { cancelSignal: signal })

      expect(systemOne).toHaveBeenCalledWith(expect.objectContaining({ model: 'jev-latest' }), { signal })
    })
  })

  it('is calibrated and exposes a mutable config', () => {
    const jev = model(200, OK_BODY)

    jev.updateConfig({ modelId: 'jev-1.14.0' })

    expect({ calibrated: jev.calibrated, config: jev.getConfig() }).toStrictEqual({
      calibrated: true,
      config: {
        modelId: 'jev-1.14.0',
        maxStatePlusQuestionTokens: 32_000,
        maxRequestTokens: 64_000,
        temperature: 1,
        calibrated: true,
      },
    })
  })

  describe('calibration', () => {
    function calibrated(options: { temperature?: number; calibrated?: boolean }): TypeSafeDecisionModel {
      return new TypeSafeDecisionModel({
        apiKey: 'k',
        ...options,
        clientConfig: { fetch: fakeFetch(200, OK_BODY), retry: { maxRetries: 0 }, logLevel: 'off' },
      })
    }

    it('passes the server numbers through at the default temperature', async () => {
      const response = await calibrated({}).ask('s', QUESTIONS)

      expect(response.answers.dept).toStrictEqual(new ChoiceAnswer('billing', { billing: 0.95, technical: 0.05 }, 0.9))
    })

    it('rescales on the client as softmax(log p / T) and keeps the server numbers on extras', async () => {
      const response = await calibrated({ temperature: 2 }).ask('s', QUESTIONS)

      const dept = response.answers.dept as ChoiceAnswer
      const urgent = response.answers.urgent as YesNoAnswer
      const severity = response.answers.severity as ScoreAnswer
      const billing = Math.sqrt(0.95) / (Math.sqrt(0.95) + Math.sqrt(0.05))
      const yes = Math.sqrt(0.8) / (Math.sqrt(0.8) + Math.sqrt(0.2))
      expect(dept.choice).toBe('billing')
      expect(dept.probabilities.billing).toBeCloseTo(billing, 12)
      expect(dept.confidence).toBeCloseTo(billing, 12)
      expect(dept.logits).toBeUndefined()
      expect(dept.extras).toStrictEqual({ confidence: 0.9, probabilities: { billing: 0.95, technical: 0.05 } })
      expect(urgent.probability).toBeCloseTo(yes, 12)
      expect(urgent.confidence).toBeCloseTo(Math.abs(2 * yes - 1), 12)
      expect(urgent.extras).toStrictEqual({ noul: 0.8 })
      const weighted = Object.entries(severity.probabilities).reduce((sum, [level, p]) => sum + Number(level) * p, 0)
      expect(severity.score).toBeCloseTo(weighted, 12)
      expect(severity.extras).toMatchObject({ score: 1.2, confidence: 0.4 })
    })

    it('reports no confidence when uncalibrated, and confidence gates refuse the instance', async () => {
      const kev = calibrated({ calibrated: false })

      const response = await kev.ask('s', QUESTIONS)

      expect(kev.calibrated).toBe(false)
      expect(response.answers.dept!.confidence).toBeUndefined()
      expect(response.answers.dept!.extras).toMatchObject({ confidence: 0.9 })
      expect(response.answers.urgent!.confidence).toBeUndefined()
      expect(() => new DecisionStrategy(kev, { minConfidence: 0.7 })).toThrow(
        'configure a raw-logit provider with temperature='
      )
    })

    it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])('rejects temperature %s', (temperature) => {
      expect(() => calibrated({ temperature })).toThrow('temperature must be a finite number above 0')
    })

    it('validates calibration settings on updateConfig', () => {
      const jev = calibrated({ temperature: 1.4 })

      jev.updateConfig({ temperature: 2.1, calibrated: false })

      expect({ temperature: jev.getConfig().temperature, calibrated: jev.calibrated }).toStrictEqual({
        temperature: 2.1,
        calibrated: false,
      })
      expect(() => jev.updateConfig({ calibrated: 'yes' as unknown as boolean })).toThrow(
        'calibrated must be a boolean'
      )
      expect(() => jev.updateConfig({ temperature: 0 })).toThrow('temperature must be')
    })
  })
})
