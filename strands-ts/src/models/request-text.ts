/**
 * Bounded, safe text projections of an agent conversation for auxiliary decision calls.
 *
 * Shared by model-routing classification and System One decision adapters so every auxiliary call bounds and
 * sanitizes the conversation the same way: only request-bearing user text crosses the boundary, guarded content is
 * never forwarded, media is labelled rather than sent, and every projection is length-bounded.
 */
import type { TextBlock } from '../types/messages.js'
import type { Message, SystemPrompt } from '../types/messages.js'

/** Marker inserted where bounded text was cut. */
export const OMISSION_MARKER = '\n...[content omitted]...\n'

/** Approximate characters per token used to turn token budgets into character bounds. */
export const CHARS_PER_TOKEN = 4

/** Placeholder when the conversation has no request-bearing user message. */
export const NO_REQUEST_TEXT = '[No request-bearing user message provided]'

/** Bounded decision state projected from an agent conversation. */
export interface ProjectedState {
  /** The latest request-bearing user message, bounded. */
  readonly request: string
  /** Text of the agent's system prompt, bounded. */
  readonly agent_instructions: string
}

/** Options for {@link projectState}. */
export interface ProjectStateOptions {
  /** Approximate token budget for the request text. */
  readonly maxTokens: number
  /** Approximate token budget for the instruction text. Defaults to `maxTokens`. */
  readonly maxInstructionTokens?: number
}

/**
 * Project an agent conversation into bounded decision state.
 *
 * Only the latest request-bearing user message and the text of the system prompt cross the boundary; guarded
 * content is labelled, never forwarded, and media is labelled rather than sent.
 *
 * @param messages - The conversation
 * @param systemPrompt - The agent's system prompt, if any
 * @param options - Token budgets for the request and instruction text
 * @returns `{ request, agent_instructions }`, each bounded to its budget
 * @throws Error if a budget is not a positive integer
 */
export function projectState(
  messages: readonly Message[],
  systemPrompt: SystemPrompt | undefined,
  options: ProjectStateOptions
): ProjectedState {
  const instructionTokens = options.maxInstructionTokens ?? options.maxTokens
  requirePositiveInteger('maxTokens', options.maxTokens)
  requirePositiveInteger('maxInstructionTokens', instructionTokens)
  return {
    request: latestRequestText(messages, options.maxTokens * CHARS_PER_TOKEN),
    agent_instructions: instructionText(systemPrompt, instructionTokens * CHARS_PER_TOKEN),
  }
}

/**
 * Return the latest request-bearing user message as bounded safe text.
 *
 * @param messages - The conversation
 * @param characterLimit - Maximum characters returned
 * @param marker - Omission marker inserted where text is cut
 * @param noRequestText - Placeholder when no user message carries a request
 * @returns Bounded request text
 */
export function latestRequestText(
  messages: readonly Message[],
  characterLimit: number,
  marker: string = OMISSION_MARKER,
  noRequestText: string = NO_REQUEST_TEXT
): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!
    if (message.role !== 'user') continue
    const request = requestText(message, characterLimit, marker)
    if (request !== undefined) return request
  }
  return truncateText(noRequestText, characterLimit, marker)
}

/**
 * Extract bounded text from an agent system prompt, omitting non-text blocks such as cache points.
 *
 * @param systemPrompt - The agent's system prompt, if any
 * @param characterLimit - Maximum characters returned
 * @param marker - Omission marker inserted where text is cut
 * @returns Bounded instruction text, empty when there is none
 */
export function instructionText(
  systemPrompt: SystemPrompt | undefined,
  characterLimit: number,
  marker: string = OMISSION_MARKER
): string {
  if (systemPrompt === undefined) return ''
  const instructions =
    typeof systemPrompt === 'string'
      ? systemPrompt
      : systemPrompt
          .filter((block): block is TextBlock => block.type === 'textBlock')
          .map((block) => block.text)
          .join('\n')
  return truncateText(instructions, characterLimit, marker)
}

/**
 * Bound text while preserving its opening and trailing content.
 *
 * @param text - Text to bound
 * @param characterLimit - Maximum characters returned
 * @param marker - Omission marker inserted where text is cut
 * @returns The text, or its head and tail around the marker
 */
export function truncateText(text: string, characterLimit: number, marker: string = OMISSION_MARKER): string {
  if (text.length <= characterLimit) return text
  if (characterLimit <= marker.length) return text.slice(0, characterLimit)
  const availableCharacters = characterLimit - marker.length
  const headCharacters = Math.floor(availableCharacters / 2)
  const tailCharacters = availableCharacters - headCharacters
  return `${text.slice(0, headCharacters)}${marker}${text.slice(-tailCharacters)}`
}

/** Render only safe request-bearing fields from one user message, or undefined when it carries no request. */
function requestText(message: Message, characterLimit: number, marker: string): string | undefined {
  const parts: string[] = []
  for (const block of message.content) {
    const part = blockText(block)
    if (part !== undefined) parts.push(part)
  }
  if (parts.length === 0) return undefined
  return truncateText(parts.join('\n'), characterLimit, marker)
}

/** Return the safe text for one content block, or undefined when it carries none. */
function blockText(block: Message['content'][number]): string | undefined {
  switch (block.type) {
    case 'textBlock':
      return block.text.trim().length > 0 ? block.text : undefined
    case 'guardContentBlock':
      return block.text !== undefined && block.text.text.trim().length > 0 ? '[Guarded content]' : undefined
    case 'imageBlock':
      return '[Image]'
    case 'documentBlock':
      return '[Document]'
    case 'videoBlock':
      return '[Video]'
    default:
      return undefined
  }
}

function requirePositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`)
}
