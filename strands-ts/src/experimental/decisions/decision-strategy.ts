/**
 * Model routing with a System One decision model.
 */
import { normalizeError } from '../../errors.js'
import { logger } from '../../logging/logger.js'
import { projectState } from '../../models/request-text.js'
import type { RoutingCandidate } from '../../models/routing/router.js'
import type { RoutingContext, RoutingStrategy } from '../../models/routing/strategy.js'
import { DecisionModel } from './decision-model.js'
import { Choice, ChoiceAnswer } from './types.js'

const DEFAULT_INSTRUCTIONS =
  'Which candidate model is the least capable one that can still fully and accurately handle `request`? ' +
  'Rule out candidates whose description shows they cannot meet a requirement of the request; reserve more ' +
  'capable candidates for requests whose complexity genuinely needs them. `agent_instructions` describe the ' +
  'agent the request is sent to. Treat missing candidate evidence as unknown, not unsupported.'

/** Options for constructing a {@link DecisionStrategy}. */
export interface DecisionStrategyOptions {
  /** Decline (serve the router default) when the choice's confidence is below this. Requires a calibrated model. */
  readonly minConfidence?: number
  /** The routing question. Candidate evidence and the request are sent as state. */
  readonly instructions?: string
  /** Token budget for the latest request text sent as state. Defaults to 1000. */
  readonly maxRequestTokens?: number
  /** Token budget for the parent agent's system-prompt text sent as state. Defaults to 1000. */
  readonly maxInstructionTokens?: number
}

/**
 * Throw when confidence gating is configured on a model whose confidence is not meaningful.
 *
 * @param adapter - Adapter name for the error message
 * @param model - The decision model
 * @param minConfidence - The configured floor, if any
 * @throws Error if `minConfidence` is set and `model` is not calibrated
 * @internal
 */
export function requireCalibrated(adapter: string, model: DecisionModel, minConfidence: number | undefined): void {
  if (minConfidence === undefined || model.calibrated) return
  throw new Error(
    `${adapter}(minConfidence=${minConfidence}) needs a calibrated DecisionModel; ${model.constructor.name} is ` +
      'not. Remove minConfidence, use a calibrated provider, or configure a raw-logit provider with temperature= ' +
      '(see fit_temperature).'
  )
}

/**
 * Choose the router's candidate with one System One `Choice` over the candidates' evidence.
 *
 * Unlike an LLM classifier, the answer carries a calibrated confidence, so this strategy can decline on
 * ambiguity as well as on errors: below `minConfidence` the router serves its default candidate. It chooses only
 * the opening candidate and declines after a failure, so the model's error surfaces.
 *
 * @example
 * ```typescript
 * const router = new ModelRouter(
 *   [
 *     new RoutingCandidate({ model: fast, name: 'routine' }),
 *     new RoutingCandidate({ model: strong, name: 'complex' }),
 *   ],
 *   { strategy: new DecisionStrategy(new TypeSafeDecisionModel(), { minConfidence: 0.7 }) }
 * )
 * ```
 */
export class DecisionStrategy implements RoutingStrategy {
  private readonly _model: DecisionModel
  private readonly _minConfidence: number | undefined
  private readonly _instructions: string
  private readonly _requestTokens: number
  private readonly _instructionTokens: number

  /**
   * Create a decision strategy.
   *
   * @param decisionModel - The model that makes the routing decision
   * @param options - Confidence floor, routing question, and token budgets
   * @throws TypeError if `decisionModel` is not a DecisionModel
   * @throws Error if `minConfidence` is set on an uncalibrated model, or a budget is not a positive integer
   */
  constructor(decisionModel: DecisionModel, options: DecisionStrategyOptions = {}) {
    if (!(decisionModel instanceof DecisionModel)) throw new TypeError('decisionModel must be a DecisionModel')
    requireCalibrated('DecisionStrategy', decisionModel, options.minConfidence)
    const requestTokens = options.maxRequestTokens ?? 1_000
    const instructionTokens = options.maxInstructionTokens ?? 1_000
    if (!Number.isInteger(requestTokens) || requestTokens <= 0) {
      throw new Error('maxRequestTokens must be a positive integer')
    }
    if (!Number.isInteger(instructionTokens) || instructionTokens <= 0) {
      throw new Error('maxInstructionTokens must be a positive integer')
    }
    this._model = decisionModel
    this._minConfidence = options.minConfidence
    this._instructions = options.instructions ?? DEFAULT_INSTRUCTIONS
    this._requestTokens = requestTokens
    this._instructionTokens = instructionTokens
  }

  /**
   * Return the chosen opening candidate, or `undefined` to decline.
   *
   * @param context - Current request and chronological routing history
   * @returns The chosen candidate, or `undefined` to decline
   */
  async select(context: RoutingContext): Promise<RoutingCandidate | undefined> {
    if (context.attempts.length > 0) return undefined
    if (context.candidates.length === 1) return context.candidates[0]

    const keys = new Map(context.candidates.map((candidate, index) => [`c${index}`, candidate]))
    const question = new Choice(
      this._instructions,
      Object.fromEntries([...keys].map(([key, candidate]) => [key, evidence(candidate)]))
    )
    const state = projectState(context.messages, context.systemPrompt, {
      maxTokens: this._requestTokens,
      maxInstructionTokens: this._instructionTokens,
    })
    let answer: ChoiceAnswer
    try {
      const response = await this._model.ask({ ...state }, { candidate: question })
      answer = response.answers.candidate as ChoiceAnswer
    } catch (error) {
      logger.warn(
        `strategy=<${this.constructor.name}>, reason=<decision_error>, error_type=<${normalizeError(error).name}> | routing declined`
      )
      return undefined
    }
    if (this._minConfidence !== undefined && (answer.confidence ?? 0) < this._minConfidence) {
      logger.debug(
        `choice=<${answer.choice}>, confidence=<${answer.confidence}>, min_confidence=<${this._minConfidence}> | routing declined on low confidence`
      )
      return undefined
    }
    return keys.get(answer.choice)
  }
}

function evidence(candidate: RoutingCandidate): string | null {
  return candidateEvidence(candidate.name, candidate.description, candidate.metadata)
}

/**
 * Render a candidate's evidence as the `Choice` option description every decision adapter sends.
 *
 * @param name - Candidate name
 * @param description - Candidate description
 * @param metadata - Candidate metadata
 * @returns JSON evidence, or null when there is none
 * @internal
 */
export function candidateEvidence(
  name: string | undefined,
  description: string | undefined,
  metadata?: Readonly<Record<string, unknown>>
): string | null {
  const fields = {
    ...(name && { name }),
    ...(description && { description }),
    ...(metadata && Object.keys(metadata).length > 0 && { metadata }),
  }
  return Object.keys(fields).length > 0 ? JSON.stringify(fields) : null
}
