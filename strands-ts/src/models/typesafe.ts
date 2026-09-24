/**
 * TypeSafe System One decision model provider (Jev, and self-hosted servers such as Kev).
 *
 * - Docs: https://docs.typesafe.ai/
 * - API: https://docs.typesafe.ai/api
 * - Kev (open, self-hosted, same API): https://github.com/jaredpalmer/kev
 *
 * Experimental: subject to change without notice.
 */
import {
  APIError,
  BadRequestError,
  RateLimitError,
  TypeSafeClient,
  UnprocessableEntityError,
  choice as vendorChoice,
  noul as vendorNoul,
  score as vendorScore,
} from '@typesafe-ai/sdk'
import type {
  ChoiceResponse,
  NoulResponse,
  Question as VendorQuestion,
  ScoreCriteria,
  ScoreResponse,
  TypeSafeClientConfig,
} from '@typesafe-ai/sdk'

import { ContextWindowOverflowError, ModelThrottledError } from '../errors.js'
import { logger } from '../logging/logger.js'
import { answerFromLogits, checkTemperature, logitsOf } from '../experimental/decisions/calibration.js'
import { DecisionModel } from '../experimental/decisions/decision-model.js'
import type { AskOptions, DecisionModelConfig } from '../experimental/decisions/decision-model.js'
import {
  Choice,
  ChoiceAnswer,
  DecisionResponse,
  Score,
  ScoreAnswer,
  YesNo,
  YesNoAnswer,
  yesNoConfidence,
} from '../experimental/decisions/types.js'
import type { Answer, DecisionState, Question } from '../experimental/decisions/types.js'

/** Environment variable holding the TypeSafe API key. */
export const API_KEY_ENV = 'TYPESAFE_API_KEY'
/** Environment variable overriding the server root. */
export const BASE_URL_ENV = 'TYPESAFE_BASE_URL'
/** TypeSafe's hosted server. */
export const DEFAULT_BASE_URL = 'https://api.typesafe.ai'
/** Default model alias. */
export const DEFAULT_MODEL_ID = 'jev-latest'
/** Jev's estimated-token budget for the whole request. */
export const MAX_REQUEST_TOKENS = 64_000
/** Jev's estimated-token budget for the state plus the longest question. */
export const MAX_STATE_PLUS_QUESTION_TOKENS = 32_000
/**
 * Kev's budget for one row of state plus one question. Kev batches questions itself, so it has no per-request cap.
 */
export const KEV_MAX_STATE_PLUS_QUESTION_TOKENS = 8_192
/** TypeSafe API limit on Choice options, checked before sending. */
export const MAX_CHOICE_OPTIONS = 255
/** TypeSafe API limit on Score levels, checked before sending. */
export const MAX_SCORE_LEVELS = 10

const LOCAL_API_KEY = 'local'
const OVERLOADED_STATUS = 529

/** Configuration for TypeSafe decision models. */
export interface TypeSafeDecisionModelConfig extends DecisionModelConfig {
  /**
   * Estimated-token budget for the state plus the longest question, checked before sending. Defaults to Jev's 32000;
   * use {@link KEV_MAX_STATE_PLUS_QUESTION_TOKENS} for Kev.
   */
  maxStatePlusQuestionTokens?: number
  /** Estimated-token budget for the whole request, or null for no cap. Defaults to Jev's 64000; Kev has no cap. */
  maxRequestTokens?: number | null
  /**
   * Applied on the client on top of the server's probabilities, as `softmax(log p / T)`. 1.0 (the default) is the
   * identity; set a fitted value for a server that returns raw probabilities or for a domain fit.
   */
  temperature?: number
  /**
   * Whether this instance's confidences are calibrated. Defaults to true; false records that the server is
   * uncalibrated and no temperature is fitted, so confidences are undefined and confidence gates refuse it.
   */
  calibrated?: boolean
}

/** Options for constructing a {@link TypeSafeDecisionModel}. */
export interface TypeSafeDecisionModelOptions extends TypeSafeDecisionModelConfig {
  /**
   * API key sent as a bearer token. For TypeSafe's server it defaults to `TYPESAFE_API_KEY`. For any other
   * `baseUrl` it is never read from `TYPESAFE_API_KEY`, so a TypeSafe key is not sent to a third-party server;
   * without one a placeholder is sent, which an open Kev server accepts.
   */
  apiKey?: string
  /** Server root. Defaults to `TYPESAFE_BASE_URL`, else `https://api.typesafe.ai`. */
  baseUrl?: string
  /** A preconfigured client; when given, `apiKey`, `baseUrl` and `clientConfig` are ignored. */
  client?: TypeSafeClient
  /** Extra client options (for example `timeout` or `retry`). */
  clientConfig?: Omit<TypeSafeClientConfig, 'apiKey' | 'baseURL' | 'defaultModel'>
}

/** Resolved configuration: every budget is set. */
interface ResolvedConfig extends TypeSafeDecisionModelConfig {
  modelId: string
  maxStatePlusQuestionTokens: number
  maxRequestTokens: number | null
  temperature: number
  calibrated: boolean
}

/**
 * System One decisions from any server that speaks TypeSafe's `/v1/systemone` API.
 *
 * By default this calls TypeSafe's hosted Jev models. Pass `baseUrl` to call a self-hosted server with the same
 * API, such as Kev.
 *
 * Calibration is part of the configured instance. Jev's probabilities are trained for calibration, and each Kev
 * checkpoint ships with a fitted temperature, so the defaults (`temperature: 1.0`, the identity, and
 * `calibrated: true`) pass the server's numbers through unchanged. For a Kev server started with
 * `KEV_TEMPERATURE=1.0` (raw probabilities), pass the temperature you fitted as `temperature`, or `calibrated: false`
 * if you have none. When the SDK recomputes an answer, the server's original numbers are kept on `answer.extras`. The
 * `/v1/systemone` wire returns probabilities, not logits, so `answer.logits` is undefined.
 *
 * Aliases such as `jev-latest` can move to a new release; pin a versioned id (for example `jev-1.13.0`) once you have
 * tuned thresholds against it. The id the server reports is recorded on every response and span.
 *
 * @example
 * ```typescript
 * import { z } from 'zod'
 * import { choice } from '@strands-agents/sdk/experimental'
 * import { TypeSafeDecisionModel } from '@strands-agents/sdk/models/typesafe'
 *
 * const jev = new TypeSafeDecisionModel({ modelId: 'jev-1.13.0' })
 * const Triage = z.object({ department: choice(['billing', 'technical']).describe('Which team owns `ticket`?') })
 * const decision = await jev.decide(Triage, { ticket: 'I was charged twice' })
 * decision.output.department // 'billing'
 * ```
 */
export class TypeSafeDecisionModel extends DecisionModel<TypeSafeDecisionModelConfig> {
  /** The server root requests are sent to. */
  readonly baseUrl: string
  private _config: ResolvedConfig
  private readonly _client: TypeSafeClient

  /**
   * Create a TypeSafe decision model.
   *
   * @param options - Model configuration and client options
   * @throws Error if no client is given, the server is TypeSafe's, and no API key is found; or `temperature` is
   * not a finite number above 0, or `calibrated` is not a boolean
   */
  constructor(options: TypeSafeDecisionModelOptions = {}) {
    super()
    const { apiKey, baseUrl, client, clientConfig, ...modelConfig } = options
    checkCalibrationConfig(modelConfig)
    this._config = {
      modelId: DEFAULT_MODEL_ID,
      maxStatePlusQuestionTokens: MAX_STATE_PLUS_QUESTION_TOKENS,
      maxRequestTokens: MAX_REQUEST_TOKENS,
      temperature: 1,
      calibrated: true,
      ...modelConfig,
    }
    this.baseUrl = (baseUrl ?? (env(BASE_URL_ENV) || DEFAULT_BASE_URL)).replace(/\/+$/, '')
    if (client !== undefined) {
      this._client = client
      return
    }
    const key = apiKey ?? this._defaultApiKey()
    if (!key) throw new Error(`TypeSafeDecisionModel needs an API key: pass apiKey or set ${API_KEY_ENV}`)
    this._client = new TypeSafeClient({ ...clientConfig, apiKey: key, baseURL: this.baseUrl })
  }

  /** The configured `calibrated` value (true by default: Jev and a normally served Kev are calibrated). */
  override get calibrated(): boolean {
    return this._config.calibrated
  }

  /**
   * Update the model configuration.
   *
   * @param config - Configuration overrides
   * @throws Error if `temperature` is not a finite number above 0, or `calibrated` is not a boolean
   */
  updateConfig(config: TypeSafeDecisionModelConfig): void {
    checkCalibrationConfig(config)
    this._config = { ...this._config, ...config }
  }

  /**
   * Return the model configuration.
   *
   * @returns The resolved configuration
   */
  getConfig(): TypeSafeDecisionModelConfig {
    return { ...this._config }
  }

  /**
   * Answer every question in one `/v1/systemone` request.
   *
   * @throws ContextWindowOverflowError if the request exceeds the configured budgets; thrown before sending
   * @throws ModelThrottledError on rate limiting (429) or overload (529)
   * @throws Error if a question exceeds the API's option or level limits (thrown before sending), or the API
   * rejects the request as invalid (400/422)
   */
  protected async _ask(
    state: DecisionState,
    questions: Readonly<Record<string, Question>>,
    options: AskOptions
  ): Promise<DecisionResponse> {
    checkLimits(questions)
    checkBudget(state, questions, this._config.maxStatePlusQuestionTokens, this._config.maxRequestTokens)
    const vendorQuestions = Object.fromEntries(
      Object.entries(questions).map(([id, question]) => [id, toVendor(question)])
    )
    let response
    try {
      response = await this._client.systemOne(
        { state, questions: vendorQuestions, model: this._config.modelId },
        options.cancelSignal ? { signal: options.cancelSignal } : undefined
      )
    } catch (error) {
      throw translateError(error)
    }
    const inputTokens = response.usage.input_tokens ?? 0
    const outputTokens = response.usage.output_tokens ?? 0
    logger.debug(
      `model=<${response.model}>, questions=<${Object.keys(questions).length}>, input_tokens=<${inputTokens}> | typesafe decision answered`
    )
    const { temperature, calibrated } = this._config
    const answers = Object.fromEntries(
      Object.entries(response.answers).map(([id, answer]) => [id, fromVendor(answer, temperature, calibrated)])
    )
    return new DecisionResponse(answers, response.model, {
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
    })
  }

  private _defaultApiKey(): string {
    return this.baseUrl === DEFAULT_BASE_URL ? env(API_KEY_ENV) : LOCAL_API_KEY
  }
}

function env(name: string): string {
  return globalThis?.process?.env?.[name]?.trim() ?? ''
}

function translateError(error: unknown): unknown {
  if (error instanceof RateLimitError) return new ModelThrottledError(error.message, { cause: error })
  if (error instanceof UnprocessableEntityError || error instanceof BadRequestError) {
    return new Error(`TypeSafe rejected the decision request: ${error.message}`, { cause: error })
  }
  if (error instanceof APIError && error.status === OVERLOADED_STATUS) {
    return new ModelThrottledError(error.message, { cause: error })
  }
  return error
}

function checkLimits(questions: Readonly<Record<string, Question>>): void {
  for (const [id, question] of Object.entries(questions)) {
    const optionCount = question instanceof Choice ? Object.keys(question.options).length : 0
    if (optionCount > MAX_CHOICE_OPTIONS) {
      throw new Error(`${id}: TypeSafe allows at most ${MAX_CHOICE_OPTIONS} Choice options, got ${optionCount}`)
    }
    if (question instanceof Score && question.levels.length > MAX_SCORE_LEVELS) {
      throw new Error(`${id}: TypeSafe allows at most ${MAX_SCORE_LEVELS} Score levels, got ${question.levels.length}`)
    }
  }
}

function estimateTokens(value: unknown): number {
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  return Math.ceil(text.length / 4)
}

function checkBudget(
  state: DecisionState,
  questions: Readonly<Record<string, Question>>,
  maxRow: number,
  maxRequest: number | null
): void {
  const stateTokens = estimateTokens(state)
  const questionTokens = Object.values(questions).map((question) => estimateTokens(toVendor(question)))
  const longest = Math.max(...questionTokens)
  if (stateTokens + longest > maxRow) {
    throw new ContextWindowOverflowError(
      `decision state plus the longest question is ~${stateTokens + longest} tokens; this model allows ` +
        `${maxRow}. Trim the state or split the question.`
    )
  }
  const total = stateTokens + questionTokens.reduce((sum, tokens) => sum + tokens, 0)
  if (maxRequest !== null && total > maxRequest) {
    throw new ContextWindowOverflowError(
      `decision request is ~${total} tokens; this model allows ${maxRequest} per request. ` +
        'Split the questions across requests.'
    )
  }
}

function toVendor(question: Question): VendorQuestion {
  if (question instanceof Choice) return vendorChoice(question.instructions, { ...question.options })
  if (question instanceof Score) {
    return vendorScore(question.instructions, [...question.levels] as unknown as ScoreCriteria)
  }
  const criteria = {
    ...(question.true !== undefined && { true: question.true }),
    ...(question.false !== undefined && { false: question.false }),
  }
  return vendorNoul(question.instructions, Object.keys(criteria).length > 0 ? criteria : undefined)
}

type VendorAnswer = NoulResponse | ChoiceResponse | ScoreResponse

interface Mapped {
  readonly answer: Answer
  readonly question: Question
  readonly distribution: Readonly<Record<string | number, number>>
  readonly originals: Readonly<Record<string, unknown>>
}

/** Map a vendor answer; at T=1 and calibrated it passes through, otherwise the SDK recomputes it. */
function fromVendor(answer: VendorAnswer, temperature: number, calibrated: boolean): Answer {
  const mapped = mapVendor(answer)
  if (temperature === 1 && calibrated) return mapped.answer
  // The wire has no logits: rescale log-probabilities, keep the server's numbers on extras, and report no logits.
  const rescaled = answerFromLogits(mapped.question, logitsOf(mapped.distribution), { temperature, calibrated })
  return withoutLogits(rescaled, mapped.originals)
}

function mapVendor(answer: VendorAnswer): Mapped {
  switch (answer.type) {
    case 'noul':
      return {
        answer: new YesNoAnswer(answer.noul, yesNoConfidence(answer.noul)),
        question: new YesNo('_'),
        distribution: { true: answer.noul, false: 1 - answer.noul },
        originals: { noul: answer.noul },
      }
    case 'choice': {
      const probabilities = { ...answer.probabilities }
      return {
        answer: new ChoiceAnswer(answer.choice, probabilities, answer.confidence),
        question: new Choice('_'),
        distribution: probabilities,
        originals: { confidence: answer.confidence, probabilities },
      }
    }
    case 'score': {
      const probabilities = Object.fromEntries(
        Object.entries(answer.probabilities).map(([level, value]) => [Number(level), value as number])
      )
      return {
        answer: new ScoreAnswer(answer.score, probabilities, answer.confidence),
        question: new Score('_', ['_', '_']),
        distribution: probabilities,
        originals: { score: answer.score, confidence: answer.confidence, probabilities: { ...answer.probabilities } },
      }
    }
    default:
      throw new Error(`unsupported TypeSafe answer type '${(answer as { type: string }).type}'`)
  }
}

function withoutLogits(answer: Answer, extras: Readonly<Record<string, unknown>>): Answer {
  if (answer instanceof ChoiceAnswer)
    return new ChoiceAnswer(answer.choice, answer.probabilities, answer.confidence, { extras })
  if (answer instanceof ScoreAnswer)
    return new ScoreAnswer(answer.score, answer.probabilities, answer.confidence, { extras })
  return new YesNoAnswer(answer.probability, answer.confidence, { extras })
}

function checkCalibrationConfig(config: TypeSafeDecisionModelConfig): void {
  if (config.temperature !== undefined) checkTemperature(config.temperature)
  if (config.calibrated !== undefined && typeof config.calibrated !== 'boolean') {
    throw new Error(`calibrated must be a boolean, got ${String(config.calibrated)}`)
  }
}
