/**
 * System-One-first action selection: a decision model serves closed-set steps before the LLM is asked.
 */
import type { AttributeValue } from '@opentelemetry/api'

import { DECISION_SOURCE, synthesizedMetadata, synthesizedSource } from '../../agent/synthesized.js'
import { normalizeError } from '../../errors.js'
import { logger } from '../../logging/logger.js'
import { InvokeModelStage } from '../../middleware/stages.js'
import type { InvokeModelContext, InvokeModelResult } from '../../middleware/stages.js'
import type { MiddlewareNext } from '../../middleware/types.js'
import { CHARS_PER_TOKEN, projectState, truncateText } from '../../models/request-text.js'
import { ModelMetadataEvent } from '../../models/streaming.js'
import type { Plugin } from '../../plugins/plugin.js'
import type { AgentStreamEvent, LocalAgent } from '../../types/agent.js'
import { deepCopy } from '../../types/json.js'
import type { JSONValue } from '../../types/json.js'
import { Message, ToolUseBlock } from '../../types/messages.js'
import type { ToolResultContent } from '../../types/messages.js'
import { DecisionModel } from './decision-model.js'
import { requireCalibrated } from './decision-strategy.js'
import { Choice, ChoiceAnswer } from './types.js'
import type { DecisionResponse } from './types.js'

/** The option the decision model picks when no configured action fits; the LLM then runs. */
export const OTHER_OPTION = 'other'

/** Prefix of every `toolUseId` the fast path synthesizes, so an audit can find them without the metadata. */
export const TOOL_USE_ID_PREFIX = 's1-'

const DEFAULT_INSTRUCTIONS =
  'Which single action in the options should the agent take next to progress `request`, given ' +
  '`agent_instructions` and `latest_tool_result`? Choose `other` when none fits, when the request is complete, or ' +
  'when the next step needs reasoning a fixed action cannot express.'

/** A fixed tool call the fast path may emit for one option. */
export class ToolCall {
  /** Name of a tool registered on the agent. */
  readonly name: string
  /** The tool input, sent verbatim (a copy per call). */
  readonly input: Readonly<Record<string, JSONValue>>
  /** What choosing this action means, shown to the decision model. Defaults to the call itself. */
  readonly description: string | undefined

  /**
   * @param name - Name of a tool registered on the agent
   * @param input - The tool input
   * @param description - What choosing this action means, shown to the decision model
   */
  constructor(name: string, input: Readonly<Record<string, JSONValue>> = {}, description?: string) {
    this.name = name
    this.input = input
    this.description = description
  }

  /** @internal */
  _optionDescription(): string {
    return this.description || JSON.stringify({ tool: this.name, input: this.input })
  }
}

/** Options for constructing a {@link FastPath}. */
export interface FastPathOptions {
  /** Serve an action only at or above this confidence. Required: an unconditional fast path drives the agent open-loop. */
  readonly minConfidence: number
  /** The question asked before each model call. */
  readonly instructions?: string
  /** Force an LLM turn after this many consecutive synthesized turns. Defaults to 3. */
  readonly maxConsecutive?: number
  /** Token budget for the request text and for the latest tool result sent as state. Defaults to 1000. */
  readonly maxRequestTokens?: number
  /** Token budget for the agent's system-prompt text sent as state. Defaults to 1000. */
  readonly maxInstructionTokens?: number
}

type DeclineReason = 'other' | 'below_floor' | 'tool_not_offered'

/**
 * Serve a model call from a System One decision model when the next step is one of a few known actions.
 *
 * Before each model call, one `Choice` over `actions` plus `other` is asked about the conversation. A confident
 * answer (at or above `minConfidence`) short-circuits the call: the agent receives a synthesized assistant turn
 * holding that action's tool call, with stop reason `toolUse`, and runs the tool normally, so tool hooks,
 * interventions and tool spans all apply. `other`, a below-floor answer, or a decision error runs the LLM unchanged.
 *
 * A synthesized turn is never passed off as model output. `message.metadata.custom.strands` records
 * `{ source: 'decision', decision_model, confidence }`, the `toolUseId` starts with `s1-`, the decision span (with
 * `strands.fast_path.action`) stands in for the cycle's `chat` span, no `AfterModelCallEvent` fires, and no LLM
 * usage is billed for the cycle.
 *
 * @example
 * ```typescript
 * const agent = new Agent({
 *   model,
 *   tools: [click, scroll],
 *   plugins: [
 *     new FastPath(new TypeSafeDecisionModel(), {
 *       click_submit: new ToolCall('click', { selector: '#submit' }),
 *       scroll_down: new ToolCall('scroll', { dy: 600 }),
 *     }, { minConfidence: 0.8 }),
 *   ],
 * })
 * ```
 */
export class FastPath implements Plugin {
  readonly name = 'strands:fast-path'

  private readonly _model: DecisionModel
  private readonly _actions: ReadonlyMap<string, ToolCall>
  private readonly _minConfidence: number
  private readonly _maxConsecutive: number
  private readonly _requestTokens: number
  private readonly _instructionTokens: number
  private readonly _question: Choice

  /**
   * @param decisionModel - A calibrated decision model
   * @param actions - Option name to the tool call it emits. `other` is reserved.
   * @param options - Confidence floor, question, and bounds
   * @throws TypeError if `decisionModel` is not a DecisionModel or an action is not a ToolCall
   * @throws Error if the model is uncalibrated, `actions` is empty or uses `other`, `minConfidence` is outside
   *   (0, 1], or a bound is not a positive integer
   */
  constructor(decisionModel: DecisionModel, actions: Readonly<Record<string, ToolCall>>, options: FastPathOptions) {
    if (!(decisionModel instanceof DecisionModel)) throw new TypeError('decisionModel must be a DecisionModel')
    requireCalibrated('FastPath', decisionModel, options.minConfidence)
    validateActions(actions)
    if (!(options.minConfidence > 0 && options.minConfidence <= 1)) {
      throw new Error('minConfidence must be greater than 0 and at most 1')
    }
    const bounds = {
      maxConsecutive: options.maxConsecutive ?? 3,
      maxRequestTokens: options.maxRequestTokens ?? 1_000,
      maxInstructionTokens: options.maxInstructionTokens ?? 1_000,
    }
    for (const [name, value] of Object.entries(bounds)) {
      if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`)
    }
    this._model = decisionModel
    this._actions = new Map(Object.entries(actions))
    this._minConfidence = options.minConfidence
    this._maxConsecutive = bounds.maxConsecutive
    this._requestTokens = bounds.maxRequestTokens
    this._instructionTokens = bounds.maxInstructionTokens
    this._question = new Choice(options.instructions ?? DEFAULT_INSTRUCTIONS, {
      ...Object.fromEntries([...this._actions].map(([option, action]) => [option, action._optionDescription()])),
      [OTHER_OPTION]: 'None of the actions fits; let the reasoning model decide',
    })
  }

  /**
   * Register the fast-path middleware on the agent's model invocation. An action whose tool is not offered on a
   * given call is never served, so tools registered later work.
   *
   * @param agent - The agent to attach to
   */
  initAgent(agent: LocalAgent): void {
    agent.addMiddleware(InvokeModelStage, (context, next) => this._middleware(context, next))
  }

  private async *_middleware(
    context: InvokeModelContext,
    next: MiddlewareNext<InvokeModelContext, InvokeModelResult, AgentStreamEvent>
  ): AsyncGenerator<AgentStreamEvent, InvokeModelResult, undefined> {
    const served = await this._serve(context)
    if (served === undefined) return yield* next(context)
    return served
  }

  private async _serve(context: InvokeModelContext): Promise<InvokeModelResult | undefined> {
    if (!this._eligible(context)) return undefined
    let chosen: string | undefined
    const started = Date.now()
    let response: DecisionResponse
    try {
      response = await this._model._tracedAsk(this._state(context), { action: this._question }, {}, (response) => {
        const outcome = this._outcome(response.answers.action as ChoiceAnswer, context)
        chosen = outcome.chosen
        return outcome.attributes
      })
    } catch (error) {
      logger.warn(
        `adapter=<FastPath>, reason=<decision_error>, error_type=<${normalizeError(error).name}> | passing through to the model`
      )
      return undefined
    }
    if (chosen === undefined) return undefined
    return this._synthesize(chosen, response.answers.action as ChoiceAnswer, response, Date.now() - started)
  }

  private _eligible(context: InvokeModelContext): boolean {
    // A forced tool choice (structured output) must reach the model.
    if (context.toolChoice !== undefined) return false
    if (consecutiveSynthesized(context.messages) >= this._maxConsecutive) {
      logger.debug(`max_consecutive=<${this._maxConsecutive}> | forcing a model turn`)
      return false
    }
    return true
  }

  private _outcome(
    answer: ChoiceAnswer,
    context: InvokeModelContext
  ): { chosen?: string; attributes: Record<string, AttributeValue> } {
    const reason = this._declineReason(answer, context)
    if (reason !== undefined) {
      logger.debug(
        `choice=<${answer.choice}>, confidence=<${answer.confidence}>, reason=<${reason}> | passing through to the model`
      )
      return { attributes: { 'strands.fast_path.served': false, 'strands.fast_path.reason': reason } }
    }
    return {
      chosen: answer.choice,
      attributes: { 'strands.fast_path.served': true, 'strands.fast_path.action': answer.choice },
    }
  }

  private _declineReason(answer: ChoiceAnswer, context: InvokeModelContext): DeclineReason | undefined {
    if (answer.choice === OTHER_OPTION) return 'other'
    if (answer.confidence === undefined || answer.confidence < this._minConfidence) return 'below_floor'
    const offered = new Set(context.toolSpecs.map((spec) => spec.name))
    if (!offered.has(this._actions.get(answer.choice)!.name)) return 'tool_not_offered'
    return undefined
  }

  private _state(context: InvokeModelContext): Record<string, string> {
    const state: Record<string, string> = {
      ...projectState(context.messages, context.systemPrompt, {
        maxTokens: this._requestTokens,
        maxInstructionTokens: this._instructionTokens,
      }),
    }
    const result = latestToolResultText(context.messages, this._requestTokens * CHARS_PER_TOKEN)
    if (result !== undefined) state.latest_tool_result = result
    return state
  }

  private _synthesize(
    option: string,
    answer: ChoiceAnswer,
    response: DecisionResponse,
    latencyMs: number
  ): InvokeModelResult {
    const action = this._actions.get(option)!
    const decisionModel = response.modelId ?? this._model.modelId ?? null
    const message = new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({
          name: action.name,
          toolUseId: `${TOOL_USE_ID_PREFIX}${globalThis.crypto.randomUUID()}`,
          input: deepCopy(action.input) as JSONValue,
        }),
      ],
      metadata: synthesizedMetadata(DECISION_SOURCE, {
        decision_model: decisionModel,
        // A served answer always cleared the floor, so it has a confidence.
        confidence: answer.confidence!,
      }),
    })
    logger.debug(`action=<${option}>, confidence=<${answer.confidence}> | fast path served the model call`)
    return {
      result: {
        message,
        stopReason: 'toolUse',
        metadata: new ModelMetadataEvent({
          type: 'modelMetadataEvent',
          usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
          metrics: { latencyMs },
        }),
      },
    }
  }
}

function validateActions(actions: Readonly<Record<string, ToolCall>>): void {
  const entries = Object.entries(actions)
  if (entries.length === 0) throw new Error('FastPath needs at least one action')
  if (Object.hasOwn(actions, OTHER_OPTION)) {
    throw new Error(`'${OTHER_OPTION}' is reserved for the pass-through option; rename that action`)
  }
  for (const [option, action] of entries) {
    if (!(action instanceof ToolCall)) throw new TypeError(`action '${option}' must be a ToolCall`)
  }
}

/** Count the trailing assistant turns the fast path synthesized, up to the last model-generated one. */
function consecutiveSynthesized(messages: readonly Message[]): number {
  let count = 0
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i]!
    if (message.role !== 'assistant') continue
    if (synthesizedSource(message) !== DECISION_SOURCE) break
    count++
  }
  return count
}

/**
 * Bounded text of the tool results in the latest message, when it carries them.
 *
 * @internal
 */
export function latestToolResultText(messages: readonly Message[], characterLimit: number): string | undefined {
  const last = messages.at(-1)
  if (last?.role !== 'user') return undefined
  const parts: string[] = []
  for (const block of last.content) {
    if (block.type !== 'toolResultBlock') continue
    if (block.status === 'error') parts.push('[error]')
    for (const item of block.content) parts.push(...resultItemText(item))
  }
  return parts.length > 0 ? truncateText(parts.join('\n'), characterLimit) : undefined
}

const MEDIA_LABELS: Readonly<Record<string, string>> = {
  imageBlock: '[Image]',
  documentBlock: '[Document]',
  videoBlock: '[Video]',
}

function resultItemText(item: ToolResultContent): string[] {
  if (item.type === 'textBlock') return [item.text]
  if (item.type === 'jsonBlock') return [JSON.stringify(item.json)]
  return [MEDIA_LABELS[item.type]!]
}
