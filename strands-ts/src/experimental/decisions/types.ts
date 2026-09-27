/**
 * Question, answer, and response types for System One decision models.
 */
import type { Usage } from '../../models/streaming.js'
import type { JSONValue } from '../../types/json.js'

/** Text, a JSON object, or a JSON array. Instructions, option descriptions, and levels all accept it. */
export type JSONContent = string | { [key: string]: JSONValue } | JSONValue[]

/** The data a decision is made about. Treated as untrusted content, never as instructions. */
export type DecisionState = JSONContent

/** Minimum number of levels a {@link Score} needs. */
export const MIN_SCORE_LEVELS = 2

/**
 * Pick one option from a defined set.
 *
 * Providers may cap the option count (TypeSafe's API allows 255) and throw from `ask` before sending.
 */
export class Choice {
  /** Discriminator for the question kind. */
  readonly kind = 'choice' as const
  /** The decision to make. Reference state fields by backticked path, e.g. `ticket.body`. */
  readonly instructions: JSONContent
  /** Option name to description, or null for an option that needs none. */
  readonly options: Readonly<Record<string, JSONContent | null>>

  /**
   * @param instructions - The decision to make
   * @param options - Option name to description, or null for an option that needs none
   * @throws Error if instructions are empty
   */
  constructor(instructions: JSONContent, options: Readonly<Record<string, JSONContent | null>> = {}) {
    requireInstructions('Choice', instructions)
    this.instructions = instructions
    this.options = options
  }
}

/**
 * Rate state along two or more ordered, self-describing levels.
 *
 * Providers may cap the level count (TypeSafe's hosted API allows 10) and throw from `ask` before sending.
 */
export class Score {
  /** Discriminator for the question kind. */
  readonly kind = 'score' as const
  /** What to rate. */
  readonly instructions: JSONContent
  /** Ordered level descriptions, lowest first. Level `i` scores `i`. */
  readonly levels: readonly JSONContent[]

  /**
   * @param instructions - What to rate
   * @param levels - Ordered level descriptions, lowest first
   * @throws Error if instructions are empty or there are fewer than two levels
   */
  constructor(instructions: JSONContent, levels: readonly JSONContent[]) {
    requireInstructions('Score', instructions)
    if (!Array.isArray(levels) || levels.length < MIN_SCORE_LEVELS) {
      throw new Error(`Score needs at least ${MIN_SCORE_LEVELS} levels`)
    }
    this.instructions = instructions
    this.levels = levels
  }
}

/** Options for a {@link YesNo} question. */
export interface YesNoOptions {
  /** Description of what a yes means. */
  readonly true?: JSONContent
  /** Description of what a no means. */
  readonly false?: JSONContent
  /** Probability at or above which a schema's boolean field reads true. Defaults to 0.5. */
  readonly threshold?: number
}

/** Ask whether a condition holds; the answer is the probability that it does. */
export class YesNo {
  /** Discriminator for the question kind. */
  readonly kind = 'yesno' as const
  /** The yes/no question. */
  readonly instructions: JSONContent
  /** Description of what a yes means. */
  readonly true?: JSONContent
  /** Description of what a no means. */
  readonly false?: JSONContent
  /** Probability at or above which a schema's boolean field reads true. The raw probability is always kept. */
  readonly threshold: number

  /**
   * @param instructions - The yes/no question
   * @param options - Outcome descriptions and the boolean threshold
   * @throws Error if instructions are empty or threshold is outside [0, 1]
   */
  constructor(instructions: JSONContent, options: YesNoOptions = {}) {
    requireInstructions('YesNo', instructions)
    const threshold = options.threshold ?? 0.5
    if (!(threshold >= 0 && threshold <= 1)) throw new Error('YesNo threshold must be between 0 and 1')
    this.instructions = instructions
    if (options.true !== undefined) this.true = options.true
    if (options.false !== undefined) this.false = options.false
    this.threshold = threshold
  }
}

/** A closed question a System One model can answer. */
export type Question = Choice | Score | YesNo

/**
 * The selected option and the distribution over every option.
 *
 * `confidence` is undefined when the model is not calibrated (see `DecisionModel.calibrated`).
 */
export class ChoiceAnswer {
  /** Discriminator for the answer kind. */
  readonly kind = 'choice' as const
  /** The selected option. */
  readonly choice: string
  /** Probability of every option. */
  readonly probabilities: Readonly<Record<string, number>>
  /**
   * Calibrated confidence in `choice`, or undefined when uncalibrated. A provider with no native confidence uses
   * `maxProbabilityConfidence(probabilities)`.
   */
  readonly confidence?: number
  /** Pre-softmax, pre-temperature scores keyed like `probabilities`, when the provider exposes them. */
  readonly logits?: Readonly<Record<string, number>>
  /** Other provider-native fields, verbatim (for example a native confidence the SDK recomputed). */
  readonly extras: Readonly<Record<string, unknown>>

  /**
   * @param choice - The selected option
   * @param probabilities - Probability of every option
   * @param confidence - How sure the model is, when calibrated
   * @param raw - Provider-native `logits` and `extras`
   * @throws Error if the distribution or logits are malformed, or the choice is not in the distribution
   */
  constructor(
    choice: string,
    probabilities: Readonly<Record<string, number>>,
    confidence?: number,
    raw: RawScores<string> = {}
  ) {
    checkDistribution(probabilities)
    checkLogits(raw.logits, probabilities)
    if (!Object.hasOwn(probabilities, choice)) {
      throw new Error(`choice '${choice}' is not one of the answered options`)
    }
    this.choice = choice
    this.probabilities = probabilities
    if (confidence !== undefined) this.confidence = confidence
    if (raw.logits !== undefined) this.logits = raw.logits
    this.extras = raw.extras ?? {}
  }
}

/**
 * The probability-weighted level and the distribution over levels.
 *
 * `probabilities` is keyed by level index, 0 to `levels.length - 1`. `score` is the probability-weighted index, so
 * it lies in `[0, levels.length - 1]`: threshold it for "how much". `confidence` ("how sure") is undefined when the
 * model is not calibrated.
 */
export class ScoreAnswer {
  /** Discriminator for the answer kind. */
  readonly kind = 'score' as const
  /** The probability-weighted level index. */
  readonly score: number
  /** Probability of every level, keyed by level index. */
  readonly probabilities: Readonly<Record<number, number>>
  /** How sure the model is, or undefined when uncalibrated. */
  readonly confidence?: number
  /** Pre-softmax, pre-temperature scores keyed like `probabilities`, when the provider exposes them. */
  readonly logits?: Readonly<Record<number, number>>
  /** Other provider-native fields, verbatim. */
  readonly extras: Readonly<Record<string, unknown>>

  /**
   * @param score - The probability-weighted level index
   * @param probabilities - Probability of every level, keyed by level index
   * @param confidence - How sure the model is, when calibrated
   * @param raw - Provider-native `logits` and `extras`
   * @throws Error if the distribution or logits are malformed
   */
  constructor(
    score: number,
    probabilities: Readonly<Record<number, number>>,
    confidence?: number,
    raw: RawScores<number> = {}
  ) {
    checkDistribution(probabilities)
    checkLogits(raw.logits, probabilities)
    this.score = score
    this.probabilities = probabilities
    if (confidence !== undefined) this.confidence = confidence
    if (raw.logits !== undefined) this.logits = raw.logits
    this.extras = raw.extras ?? {}
  }

  /** The most probable level index. */
  get level(): number {
    let best = Number.NaN
    let bestProbability = -1
    for (const [level, probability] of Object.entries(this.probabilities)) {
      if (probability > bestProbability) {
        best = Number(level)
        bestProbability = probability
      }
    }
    return best
  }
}

/**
 * The probability that the condition holds.
 *
 * A yes/no answer's uncertainty is its probability, so policy should threshold `probability`. `confidence` exists so
 * gates that read confidence compose over every answer type: calibrated providers set it to
 * `yesNoConfidence(probability)` (0 at 0.5, 1 at 0 or 1), and it is undefined when the model is not calibrated.
 */
export class YesNoAnswer {
  /** Discriminator for the answer kind. */
  readonly kind = 'yesno' as const
  /** Probability that the condition holds. */
  readonly probability: number
  /** Distance of the probability from 0.5, scaled to [0, 1], or undefined when uncalibrated. */
  readonly confidence?: number
  /** Pre-softmax, pre-temperature scores keyed `true`/`false`, when the provider exposes them. */
  readonly logits?: Readonly<Record<YesNoKey, number>>
  /** Other provider-native fields, verbatim. */
  readonly extras: Readonly<Record<string, unknown>>

  /**
   * @param probability - Probability that the condition holds
   * @param confidence - Confidence, when calibrated
   * @param raw - Provider-native `logits` (keyed `true`/`false`) and `extras`
   * @throws Error if the probability is outside [0, 1], or the logits are malformed
   */
  constructor(probability: number, confidence?: number, raw: RawScores<YesNoKey> = {}) {
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new Error('YesNo probability must be between 0 and 1')
    }
    checkLogits(raw.logits, { true: 0, false: 0 })
    this.probability = probability
    if (confidence !== undefined) this.confidence = confidence
    if (raw.logits !== undefined) this.logits = raw.logits
    this.extras = raw.extras ?? {}
  }
}

/** A typed answer to one question. */
export type Answer = ChoiceAnswer | ScoreAnswer | YesNoAnswer

/** Logit keys of a yes/no answer. */
export type YesNoKey = 'true' | 'false'

/** Provider-native scores an answer may carry. */
export interface RawScores<K extends string | number> {
  /** Pre-softmax, pre-temperature scores keyed like the answer's probabilities. */
  readonly logits?: Readonly<Record<K, number>>
  /** Other provider-native fields, verbatim. */
  readonly extras?: Readonly<Record<string, unknown>>
}

/**
 * Confidence of a calibrated yes/no probability: its distance from 0.5, scaled to [0, 1].
 *
 * @param probability - The yes probability
 * @returns `|2p - 1|`
 */
export function yesNoConfidence(probability: number): number {
  return Math.abs(2 * probability - 1)
}

/**
 * Default confidence of a Choice or Score answer from a provider with no native confidence: the probability of the
 * most probable option or level.
 *
 * @param probabilities - The answer's distribution
 * @returns `max(probabilities)`
 * @throws Error if `probabilities` is empty
 */
export function maxProbabilityConfidence(probabilities: Readonly<Record<string | number, number>>): number {
  const values = Object.values(probabilities)
  if (values.length === 0) throw new Error('maxProbabilityConfidence needs at least one probability')
  return Math.max(...values)
}

/** Answers keyed like the questions, plus the model that answered and token usage. */
export class DecisionResponse {
  /** Answers keyed by question id. */
  readonly answers: Readonly<Record<string, Answer>>
  /** The model that answered. */
  readonly modelId?: string
  /** Token usage for the request. */
  readonly usage: Usage

  /**
   * @param answers - Answers keyed by question id
   * @param modelId - The model that answered
   * @param usage - Token usage for the request
   */
  constructor(answers: Readonly<Record<string, Answer>>, modelId?: string, usage: Usage = emptyUsage()) {
    this.answers = answers
    if (modelId !== undefined) this.modelId = modelId
    this.usage = usage
  }
}

/** A typed decision: the schema output plus the full answer for every field. */
export interface Decision<T> {
  /** The validated schema output (enum field: selected option, boolean: probability at or above threshold, score: score). */
  readonly output: T
  /** Every field's full answer, including probabilities and confidence. */
  readonly answers: Readonly<Record<string, Answer>>
  /** The model that answered. */
  readonly modelId?: string
  /** Token usage for the request. */
  readonly usage: Usage
}

/**
 * A zero-token usage record.
 *
 * @returns Usage with every count at zero
 */
export function emptyUsage(): Usage {
  return { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
}

function requireInstructions(kind: string, instructions: unknown): void {
  const isContent =
    (typeof instructions === 'string' && instructions.length > 0) ||
    (typeof instructions === 'object' && instructions !== null)
  if (!isContent) throw new Error(`${kind} instructions must be a non-empty string, JSON object, or JSON array`)
}

function checkLogits(
  logits: Readonly<Record<string | number, number>> | undefined,
  probabilities: Readonly<Record<string | number, number>>
): void {
  if (logits === undefined) return
  const logitKeys = Object.keys(logits).sort()
  const probabilityKeys = Object.keys(probabilities).sort()
  const sameKeys =
    logitKeys.length === probabilityKeys.length && logitKeys.every((key, index) => key === probabilityKeys[index])
  if (!sameKeys || Object.values(logits).some((value) => !Number.isFinite(value))) {
    throw new Error('answer logits must be finite and keyed exactly like the probabilities')
  }
}

function checkDistribution(probabilities: Readonly<Record<string | number, number>>): void {
  const values = Object.values(probabilities)
  if (values.length === 0 || values.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error('answer probabilities must be a non-empty map of finite, non-negative numbers')
  }
}
