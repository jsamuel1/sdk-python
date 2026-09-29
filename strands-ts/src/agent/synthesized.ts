/**
 * Marking for assistant turns that the SDK synthesized rather than a model generated.
 *
 * A synthesized turn is appended to the conversation like any assistant turn, because the next model call needs the
 * matching `toolUse`/`toolResult` pair. The marking lives in `MessageMetadata`, which is never sent to model
 * providers and is persisted by session managers, so replays and audits can tell it apart from generation.
 *
 * @internal
 */
import type { Message, MessageMetadata } from '../types/messages.js'
import type { JSONValue } from '../types/json.js'

/** Key under `message.metadata.custom` that holds the synthesized-turn marking. */
export const SYNTHESIZED_KEY = 'strands'

/** Source of a turn a System One decision model chose (see the experimental `FastPath`). */
export const DECISION_SOURCE = 'decision'

/**
 * Metadata recording that `source` produced a message, with any attribution `fields`.
 *
 * @param source - What produced the message
 * @param fields - Attribution recorded beside the source
 * @returns Message metadata carrying the marking
 */
export function synthesizedMetadata(source: string, fields: Readonly<Record<string, JSONValue>> = {}): MessageMetadata {
  return { custom: { [SYNTHESIZED_KEY]: { source, ...fields } } }
}

/**
 * The source that synthesized `message`, or `undefined` when a model generated it.
 *
 * @param message - The message to inspect
 * @returns The synthesizing source, if any
 */
export function synthesizedSource(message: Message): string | undefined {
  const marker = message.metadata?.custom?.[SYNTHESIZED_KEY]
  if (marker === null || typeof marker !== 'object' || Array.isArray(marker)) return undefined
  const source = (marker as Record<string, JSONValue>).source
  return typeof source === 'string' ? source : undefined
}
