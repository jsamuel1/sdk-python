/**
 * The DecisionModel abstraction: a model that decides rather than generates.
 */
import type { AttributeValue } from '@opentelemetry/api'
import type { z } from 'zod'

import { Tracer } from '../../telemetry/tracer.js'
import { normalizeError } from '../../errors.js'
import { answerFromLogits } from './calibration.js'
import { compileSchema } from './schema.js'
import type { DecisionSchema } from './schema.js'
import { Choice, ChoiceAnswer, ScoreAnswer, YesNoAnswer } from './types.js'
import type { Answer, Decision, DecisionResponse, DecisionState, Question } from './types.js'

/** Per-call options every decision model accepts. */
export interface AskOptions {
  /** Cancellation signal for the request, honored provider-dependently. */
  readonly cancelSignal?: AbortSignal
}

/** Base configuration for decision models. */
export interface DecisionModelConfig {
  /** The model identifier, recorded on decision spans. */
  modelId?: string
}

const ANSWER_TYPES = { choice: ChoiceAnswer, score: ScoreAnswer, yesno: YesNoAnswer } as const

/**
 * Abstract base for System One decision models.
 *
 * A `Model` generates: messages in, a stream of text and tool calls out. A `DecisionModel` decides: state plus
 * typed closed questions in, one typed answer per question out, with probabilities. Subclasses implement `_ask`;
 * callers use {@link DecisionModel.ask} for record questions or {@link DecisionModel.decide} for a Zod schema.
 */
export abstract class DecisionModel<TConfig extends DecisionModelConfig = DecisionModelConfig> {
  private _tracer: Tracer | undefined

  /** Turn one question's logits into an answer at a temperature; see {@link answerFromLogits}. */
  static readonly answerFromLogits = answerFromLogits

  /**
   * True when this instance's confidences are calibrated for the domain it was configured for.
   *
   * Adapters that gate on `confidence` require a calibrated model and refuse one that is not. A provider that emits
   * raw logits is calibrated only when it is configured with a fitted `temperature`.
   */
  get calibrated(): boolean {
    return false
  }

  /** Identifier of the configured model, recorded on decision spans. */
  get modelId(): string | undefined {
    return this.getConfig().modelId
  }

  /** Return the model configuration. */
  abstract getConfig(): TConfig

  /** Update the model configuration. */
  abstract updateConfig(config: TConfig): void

  /**
   * Answer every question about `state` in a single request. Implemented by providers.
   *
   * @param state - The data to decide about
   * @param questions - Question id to question
   * @param options - Per-call options
   */
  protected abstract _ask(
    state: DecisionState,
    questions: Readonly<Record<string, Question>>,
    options: AskOptions
  ): Promise<DecisionResponse>

  /**
   * Answer every question about `state`; all questions are asked together in one request.
   *
   * @param state - The data to decide about. Treated as untrusted content, never as instructions.
   * @param questions - Question id to `Choice`, `Score`, or `YesNo`. Ids are for code only.
   * @param options - Per-call options
   * @returns The answers keyed like `questions`, with the answering model id and usage
   * @throws Error if `questions` is empty, a Choice has no options, or the provider's answers do not match the questions
   */
  async ask(
    state: DecisionState,
    questions: Readonly<Record<string, Question>>,
    options: AskOptions = {}
  ): Promise<DecisionResponse> {
    return this._tracedAsk(state, questions, options)
  }

  /**
   * `ask`, plus adapter-owned attributes derived from the response and recorded on the decision span.
   *
   * @param state - The data to decide about
   * @param questions - Question id to question
   * @param options - Per-call options
   * @param spanAttributes - Derives extra decision-span attributes from the validated response
   * @returns The answers keyed like `questions`
   * @internal
   */
  async _tracedAsk(
    state: DecisionState,
    questions: Readonly<Record<string, Question>>,
    options: AskOptions = {},
    spanAttributes?: (response: DecisionResponse) => Readonly<Record<string, AttributeValue>>
  ): Promise<DecisionResponse> {
    const ids = Object.keys(questions)
    if (ids.length === 0) throw new Error('ask() needs at least one question')
    const empty = ids.filter((id) => {
      const question = questions[id]
      return question instanceof Choice && Object.keys(question.options).length === 0
    })
    if (empty.length > 0) {
      throw new Error(
        `Choice questions [${empty.join(', ')}] have no options; options may only be omitted on schema markers`
      )
    }
    const tracer = (this._tracer ??= new Tracer())
    const span = tracer.startDecisionSpan({
      modelId: this.modelId ?? this.constructor.name,
      questions: ids.map((id) => `${id}:${questions[id]!.kind}`),
    })
    let response: DecisionResponse
    try {
      response = await this._ask(state, questions, options)
      checkResponse(questions, response)
    } catch (error) {
      tracer.endDecisionSpan(span, { error: normalizeError(error) })
      throw error
    }
    tracer.endDecisionSpan(span, {
      usage: response.usage,
      answers: Object.entries(response.answers).map(([id, answer]) => summarize(id, answer)),
      ...(response.modelId !== undefined && { responseModelId: response.modelId }),
      ...(spanAttributes && { attributes: spanAttributes(response) }),
    })
    return response
  }

  /**
   * Ask every field of `schema` about `state` in one request and return a typed decision.
   *
   * @param schema - A Zod object whose fields are enums (Choice), booleans (YesNo), or `score()` numbers
   * @param state - The data to decide about
   * @param options - Per-call options, forwarded to `ask`
   * @returns The parsed schema output plus every field's full answer
   * @throws TypeError if `schema` has a field a System One model cannot answer
   * @throws Error if the provider's answers do not fit the schema
   */
  async decide<S extends DecisionSchema>(
    schema: S,
    state: DecisionState,
    options: AskOptions = {}
  ): Promise<Decision<z.infer<S>>> {
    const compiled = compileSchema(schema)
    const response = await this.ask(state, compiled.questions, options)
    return {
      output: compiled.buildOutput(response.answers),
      answers: response.answers,
      usage: response.usage,
      ...(response.modelId !== undefined && { modelId: response.modelId }),
    }
  }
}

function checkResponse(questions: Readonly<Record<string, Question>>, response: DecisionResponse): void {
  const missing = Object.keys(questions)
    .filter((id) => !Object.hasOwn(response.answers, id))
    .sort()
  if (missing.length > 0) throw new Error(`decision model returned no answer for [${missing.join(', ')}]`)
  for (const [id, question] of Object.entries(questions)) {
    const answer = response.answers[id]!
    const expected = ANSWER_TYPES[question.kind]
    if (!(answer instanceof expected)) {
      throw new Error(`${id}: expected ${expected.name}, got ${answer.constructor.name}`)
    }
    if (question instanceof Choice && !Object.hasOwn(question.options, (answer as ChoiceAnswer).choice)) {
      throw new Error(`${id}: answer '${(answer as ChoiceAnswer).choice}' is not one of the options`)
    }
  }
}

function summarize(id: string, answer: Answer): string {
  if (answer instanceof ChoiceAnswer) return `${id}=${answer.choice}@${fmt(answer.confidence)}`
  if (answer instanceof ScoreAnswer) return `${id}=${answer.score.toFixed(3)}@${fmt(answer.confidence)}`
  return `${id}=${answer.probability.toFixed(3)}`
}

function fmt(value: number | undefined): string {
  return value === undefined ? 'na' : value.toFixed(3)
}
