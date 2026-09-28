/**
 * Swarm handoff with a System One decision model.
 */
import { normalizeError } from '../../errors.js'
import { logger } from '../../logging/logger.js'
import { CHARS_PER_TOKEN, projectState, truncateText } from '../../models/request-text.js'
import type { AgentNode } from '../../multiagent/nodes.js'
import type { NodeResult } from '../../multiagent/state.js'
import type { MultiAgentInput } from '../../multiagent/multiagent.js'
import type { Handoff, HandoffContext, HandoffStrategy } from '../../multiagent/swarm.js'
import type { JSONValue } from '../../types/json.js'
import type { ContentBlock, SystemPrompt } from '../../types/messages.js'
import { Message, TextBlock, contentBlockFromData } from '../../types/messages.js'
import { DecisionModel } from './decision-model.js'
import { candidateEvidence, requireCalibrated } from './decision-strategy.js'
import { Choice, ChoiceAnswer } from './types.js'
import type { DecisionResponse } from './types.js'

/** The option that ends the swarm; a handoff decision's `next` of this value means no handoff. */
export const COMPLETE_OPTION = 'complete'

/** Key under which a swarm's `state.app` carries each node's latest handoff decision, keyed by node id. */
export const DECISION_STATE_KEY = 'decision'

/** Answer key of the handoff decision. */
const HANDOFF_FIELD = 'next'

const DEFAULT_INSTRUCTIONS =
  '`request` is the task given to the agent that just ran, `agent_instructions` describe that agent, and ' +
  '`response` is its final answer. Which candidate agent should act next? Choose `complete` when `response` ' +
  'fully resolves `request` and no other agent is needed. Treat missing candidate evidence as unknown.'
const COMPLETE_EVIDENCE = 'The task is finished: `response` resolves `request`, so no other agent should act.'
const STAY_MESSAGE =
  'A decision model could not choose the next agent with enough confidence, so you continue. Finish the task, ' +
  'or hand off if another agent should take over.'

/** The output of a handoff decision: the next node's id, or `'complete'`. */
export interface HandoffDecision {
  readonly next: string
}

/** Options for constructing a {@link DecisionHandoffStrategy}. */
export interface DecisionHandoffStrategyOptions {
  /** Apply `fallback` when the choice's confidence is below this. Requires a calibrated model. */
  readonly minConfidence?: number
  /** What to do when the decision is below the floor or fails. Defaults to `'complete'`. */
  readonly fallback?: 'complete' | 'stay'
  /** The handoff question. Candidate evidence, the request and the response are sent as state. */
  readonly instructions?: string
  /** Token budget for the node's request text sent as state. Defaults to 1000. */
  readonly maxRequestTokens?: number
  /** Token budget for the node's system-prompt text sent as state. Defaults to 1000. */
  readonly maxInstructionTokens?: number
  /** Token budget for the node's final response, sent as state and in the handoff message. Defaults to 1000. */
  readonly maxResponseTokens?: number
}

/**
 * Choose a swarm's next node with one System One `Choice` over the other nodes plus `'complete'`.
 *
 * Runs only when a node finished without handing off itself; an agent's own handoff always wins. The candidates'
 * evidence is their id and description, projected exactly as {@link DecisionStrategy} projects router candidates.
 * Each node's latest decision is recorded on the swarm's `state.app` under `'decision'`, keyed by node id.
 *
 * Below `minConfidence`, or when the decision request fails, `fallback` applies: `'complete'` ends the swarm (the
 * behaviour without a strategy) and `'stay'` re-runs the node with a note that routing was unsure. A re-run counts
 * against `maxSteps` like any handoff.
 *
 * @example
 * ```typescript
 * const swarm = new Swarm({
 *   nodes: [triage, billing, technical],
 *   handoffStrategy: new DecisionHandoffStrategy(new TypeSafeDecisionModel(), { minConfidence: 0.7 }),
 * })
 * ```
 */
export class DecisionHandoffStrategy implements HandoffStrategy {
  private readonly _model: DecisionModel
  private readonly _minConfidence: number | undefined
  private readonly _fallback: 'complete' | 'stay'
  private readonly _instructions: string
  private readonly _requestTokens: number
  private readonly _instructionTokens: number
  private readonly _responseCharacters: number

  /**
   * Create a handoff strategy.
   *
   * @param decisionModel - The model that makes the handoff decision
   * @param options - Confidence floor, fallback, handoff question, and token budgets
   * @throws TypeError if `decisionModel` is not a DecisionModel
   * @throws Error if `minConfidence` is set on an uncalibrated model, `fallback` is unknown, or a budget is not a
   *   positive integer
   */
  constructor(decisionModel: DecisionModel, options: DecisionHandoffStrategyOptions = {}) {
    if (!(decisionModel instanceof DecisionModel)) throw new TypeError('decisionModel must be a DecisionModel')
    requireCalibrated('DecisionHandoffStrategy', decisionModel, options.minConfidence)
    const fallback = options.fallback ?? 'complete'
    if (fallback !== 'complete' && fallback !== 'stay') {
      throw new Error(`fallback must be 'complete' or 'stay', got '${String(fallback)}'`)
    }
    this._model = decisionModel
    this._minConfidence = options.minConfidence
    this._fallback = fallback
    this._instructions = options.instructions ?? DEFAULT_INSTRUCTIONS
    this._requestTokens = tokenBudget('maxRequestTokens', options.maxRequestTokens)
    this._instructionTokens = tokenBudget('maxInstructionTokens', options.maxInstructionTokens)
    this._responseCharacters = tokenBudget('maxResponseTokens', options.maxResponseTokens) * CHARS_PER_TOKEN
  }

  /**
   * Return the handoff the decision model chose, the fallback, or `undefined` to complete the swarm.
   *
   * @param context - The node that ran, the candidates, and the swarm state
   * @returns The handoff, or undefined to complete
   * @throws Error if a candidate node's id is `'complete'`
   */
  async select(context: HandoffContext): Promise<Handoff | undefined> {
    if (context.candidates.length === 0) return undefined
    const keys = optionKeys(context.candidates)
    const responseText = truncateText(responseOf(context.result), this._responseCharacters)
    const question = new Choice(
      this._instructions,
      Object.fromEntries(
        [...keys].map(([key, node]) => [
          key,
          node === undefined ? COMPLETE_EVIDENCE : candidateEvidence(node.id, node.config.description),
        ])
      )
    )
    const state = {
      ...projectState([inputMessage(context.input)], systemPromptOf(context.current), {
        maxTokens: this._requestTokens,
        maxInstructionTokens: this._instructionTokens,
      }),
      response: responseText,
    }
    let response: DecisionResponse
    try {
      response = await this._model.ask(state, { [HANDOFF_FIELD]: question })
    } catch (error) {
      logger.warn(
        `strategy=<${this.constructor.name}>, node_id=<${context.current.id}>, fallback=<${this._fallback}>, error_type=<${normalizeError(error).name}> | handoff decision failed, using fallback`
      )
      return this._fallbackHandoff(context, responseText)
    }
    const answer = byNodeId(response.answers[HANDOFF_FIELD] as ChoiceAnswer, keys)
    record(context, answer, response)
    if (this._minConfidence !== undefined && (answer.confidence ?? 0) < this._minConfidence) {
      logger.debug(
        `choice=<${answer.choice}>, confidence=<${answer.confidence}>, min_confidence=<${this._minConfidence}>, fallback=<${this._fallback}> | handoff decision below floor`
      )
      return this._fallbackHandoff(context, responseText)
    }
    if (answer.choice === COMPLETE_OPTION) return undefined
    const target = context.candidates.find((node) => node.id === answer.choice)!
    const confidence = answer.confidence === undefined ? '' : ` (confidence ${answer.confidence.toFixed(2)})`
    const routed = `${context.current.id} finished without handing off; a decision model routed the task to you`
    return handoff(target, `${routed}${confidence}.`, context.current, responseText)
  }

  private _fallbackHandoff(context: HandoffContext, responseText: string): Handoff | undefined {
    if (this._fallback === 'complete') return undefined
    return handoff(context.current, STAY_MESSAGE, context.current, responseText)
  }
}

function tokenBudget(name: string, value: number = 1_000): number {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`)
  return value
}

function optionKeys(candidates: readonly AgentNode[]): Map<string, AgentNode | undefined> {
  const keys = new Map<string, AgentNode | undefined>()
  for (const [index, node] of candidates.entries()) {
    if (node.id === COMPLETE_OPTION) {
      throw new Error(`node_id=<${COMPLETE_OPTION}> | collides with the handoff decision's complete option`)
    }
    keys.set(`c${index}`, node)
  }
  keys.set(COMPLETE_OPTION, undefined)
  return keys
}

/** Re-key a Choice answer from the option keys sent to the model to node ids (and `'complete'`). */
function byNodeId(answer: ChoiceAnswer, keys: ReadonlyMap<string, AgentNode | undefined>): ChoiceAnswer {
  const name = (key: string): string => keys.get(key)?.id ?? COMPLETE_OPTION
  const rekey = (values: Readonly<Record<string, number>>): Record<string, number> =>
    Object.fromEntries(Object.entries(values).map(([key, value]) => [name(key), value]))
  return new ChoiceAnswer(name(answer.choice), rekey(answer.probabilities), answer.confidence, {
    ...(answer.logits !== undefined && { logits: rekey(answer.logits) }),
    extras: answer.extras,
  })
}

/** Store the decision as JSON: `state.app` is persisted with the session, so answer instances cannot live there. */
function record(context: HandoffContext, answer: ChoiceAnswer, response: DecisionResponse): void {
  const decision = {
    output: { next: answer.choice },
    answers: {
      [HANDOFF_FIELD]: {
        kind: answer.kind,
        choice: answer.choice,
        probabilities: { ...answer.probabilities },
        ...(answer.confidence !== undefined && { confidence: answer.confidence }),
        ...(answer.logits !== undefined && { logits: { ...answer.logits } }),
        extras: answer.extras as JSONValue,
      },
    },
    ...(response.modelId !== undefined && { modelId: response.modelId }),
    usage: { ...response.usage },
  }
  const app = context.state.app
  const recorded = (app.get(DECISION_STATE_KEY) as Record<string, JSONValue> | undefined) ?? {}
  app.set(DECISION_STATE_KEY, { ...recorded, [context.current.id]: decision })
}

/** Build a handoff whose message carries the previous node's response, which the next node cannot otherwise see. */
function handoff(target: AgentNode, message: string, previous: AgentNode, responseText: string): Handoff {
  const text = responseText ? `${message}\n\n${previous.id}'s final response:\n${responseText}` : message
  return { node: target, message: text }
}

/** A swarm node answers in its structured output's `message`; content text covers custom nodes without one. */
function responseOf(result: NodeResult): string {
  const message = (result.structuredOutput as { message?: unknown } | undefined)?.message
  if (typeof message === 'string') return message
  return result.content
    .filter((block): block is TextBlock => block.type === 'textBlock')
    .map((block) => block.text)
    .join('\n')
}

function inputMessage(input: MultiAgentInput): Message {
  if (typeof input === 'string') return new Message({ role: 'user', content: [new TextBlock(input)] })
  const content: ContentBlock[] = []
  for (const item of input) {
    if ('type' in item && item.type === 'interruptResponseContent') continue
    if ('interruptResponse' in item) continue
    content.push('type' in item ? (item as ContentBlock) : contentBlockFromData(item))
  }
  return new Message({ role: 'user', content })
}

function systemPromptOf(node: AgentNode): SystemPrompt | undefined {
  const agent = node.agent as { systemPrompt?: SystemPrompt }
  return agent.systemPrompt
}
