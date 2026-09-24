/**
 * Temperature arithmetic shared by every provider: logits (or log-probabilities) to calibrated answers.
 *
 * A provider returns scores; this module turns them into `probabilities`, `score` and `confidence` at a temperature,
 * so no vendor reimplements scaling. Rescaling log-probabilities, `softmax(log p / T)`, is the same as rescaling the
 * logits they came from, so a provider whose wire returns only probabilities uses {@link logitsOf} first.
 */
import {
  Choice,
  ChoiceAnswer,
  Score,
  ScoreAnswer,
  YesNoAnswer,
  maxProbabilityConfidence,
  yesNoConfidence,
} from './types.js'
import type { Answer, Question } from './types.js'

/** log(0) is -Infinity; zero probabilities are floored so a rescale stays finite. */
const LOG_FLOOR = 1e-12

/** Options for {@link answerFromLogits}. */
export interface AnswerFromLogitsOptions {
  /** Temperature applied as `softmax(logits / temperature)`. Defaults to 1.0, the identity. */
  readonly temperature?: number
  /** Whether the instance is calibrated. When false every confidence is undefined. Defaults to true. */
  readonly calibrated?: boolean
  /** Provider-native fields kept on the answer verbatim. */
  readonly extras?: Readonly<Record<string, unknown>>
}

/**
 * Return `temperature` if it is a finite number above zero.
 *
 * @param temperature - The candidate temperature
 * @returns The temperature
 * @throws Error if it is not a finite number above zero
 */
export function checkTemperature(temperature: unknown): number {
  if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature <= 0) {
    throw new Error(`temperature must be a finite number above 0, got ${String(temperature)}`)
  }
  return temperature
}

/**
 * Log-probabilities, usable as logits: softmax recovers the (normalized) distribution.
 *
 * @param probabilities - A distribution
 * @returns `log(max(p, 1e-12))` per key
 */
export function logitsOf<K extends string | number>(probabilities: Readonly<Record<K, number>>): Record<K, number> {
  return mapValues(probabilities, (value) => Math.log(Math.max(value, LOG_FLOOR)))
}

/**
 * `softmax(logits / temperature)`, computed stably.
 *
 * @param logits - Scores per key
 * @param temperature - Temperature; defaults to 1.0
 * @returns Probabilities per key
 */
export function softmax<K extends string | number>(
  logits: Readonly<Record<K, number>>,
  temperature = 1
): Record<K, number> {
  const scaled = mapValues(logits, (value) => value / temperature)
  const top = Math.max(...(Object.values(scaled) as number[]))
  const weights = mapValues(scaled, (value) => Math.exp(value - top))
  const total = (Object.values(weights) as number[]).reduce((sum, weight) => sum + weight, 0)
  return mapValues(weights, (weight) => weight / total)
}

/**
 * Build an answer from per-option (Choice), per-level (Score) or `true`/`false` (YesNo) logits.
 *
 * Probabilities are `softmax(logits / temperature)`. Confidence uses the default derivation:
 * {@link maxProbabilityConfidence} for Choice and Score, {@link yesNoConfidence} for YesNo, or undefined when the
 * instance is not calibrated. `logits` and `extras` are kept on the answer unchanged.
 *
 * @param question - The question the logits answer
 * @param logits - Scores keyed by option name, level index, or `true`/`false`
 * @param options - Temperature, calibration, and extras
 * @returns The answer
 * @throws Error if `temperature` is not a finite number above 0, or the logits do not match the question
 */
export function answerFromLogits(
  question: Question,
  logits: Readonly<Record<string | number, number>>,
  options: AnswerFromLogitsOptions = {}
): Answer {
  const temperature = checkTemperature(options.temperature ?? 1)
  if (Object.keys(logits).length === 0) throw new Error('answerFromLogits needs at least one logit')
  const calibrated = options.calibrated ?? true
  const raw = { logits: { ...logits }, extras: { ...options.extras } }
  const probabilities = softmax(logits, temperature)
  if (question instanceof Choice) return choiceFromProbabilities(probabilities, calibrated, raw)
  if (question instanceof Score) return scoreFromProbabilities(probabilities, calibrated, raw)
  return yesNoFromProbabilities(probabilities, calibrated, raw)
}

interface Raw {
  readonly logits: Record<string | number, number>
  readonly extras: Record<string, unknown>
}

function choiceFromProbabilities(probabilities: Record<string, number>, calibrated: boolean, raw: Raw): ChoiceAnswer {
  const confidence = calibrated ? maxProbabilityConfidence(probabilities) : undefined
  return new ChoiceAnswer(argmax(probabilities), probabilities, confidence, raw)
}

function scoreFromProbabilities(probabilities: Record<string, number>, calibrated: boolean, raw: Raw): ScoreAnswer {
  const levels = Object.fromEntries(Object.entries(probabilities).map(([level, p]) => [Number(level), p]))
  const score = Object.entries(levels).reduce((sum, [level, p]) => sum + Number(level) * p, 0)
  const confidence = calibrated ? maxProbabilityConfidence(levels) : undefined
  return new ScoreAnswer(score, levels, confidence, raw)
}

function yesNoFromProbabilities(probabilities: Record<string, number>, calibrated: boolean, raw: Raw): YesNoAnswer {
  const probability = probabilities.true
  if (probability === undefined) throw new Error("YesNo logits must be keyed 'true' and 'false'")
  const logits = raw.logits as Record<'true' | 'false', number>
  return new YesNoAnswer(probability, calibrated ? yesNoConfidence(probability) : undefined, {
    logits,
    extras: raw.extras,
  })
}

function argmax(probabilities: Readonly<Record<string, number>>): string {
  return Object.entries(probabilities).reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0]
}

function mapValues<K extends string | number>(
  record: Readonly<Record<K, number>>,
  map: (value: number) => number
): Record<K, number> {
  return Object.fromEntries(Object.entries(record).map(([key, value]) => [key, map(value as number)])) as Record<
    K,
    number
  >
}
