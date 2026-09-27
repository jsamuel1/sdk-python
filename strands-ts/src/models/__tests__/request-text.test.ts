import { describe, expect, it } from 'vitest'

import { ImageBlock } from '../../types/media.js'
import { CachePointBlock, GuardContentBlock, Message, TextBlock } from '../../types/messages.js'
import { NO_REQUEST_TEXT, OMISSION_MARKER, projectState, truncateText } from '../request-text.js'

function user(...content: Message['content']): Message {
  return new Message({ role: 'user', content })
}

describe('projectState', () => {
  it('projects the latest request-bearing user message and the system prompt text', () => {
    const messages = [
      user(new TextBlock('first')),
      new Message({ role: 'assistant', content: [new TextBlock('reply')] }),
      user(new TextBlock('latest'), new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1]) } })),
    ]

    const state = projectState(messages, [new TextBlock('Be precise'), new CachePointBlock({ cacheType: 'default' })], {
      maxTokens: 100,
    })

    expect(state).toStrictEqual({ request: 'latest\n[Image]', agent_instructions: 'Be precise' })
  })

  it('labels guarded content instead of forwarding it', () => {
    const guarded = new GuardContentBlock({ text: { text: 'secret', qualifiers: ['guard_content'] } })

    expect(projectState([user(guarded)], undefined, { maxTokens: 10 }).request).toBe('[Guarded content]')
  })

  it('falls back to a placeholder when no user message carries a request', () => {
    expect(projectState([user(new TextBlock('  '))], undefined, { maxTokens: 100 })).toStrictEqual({
      request: NO_REQUEST_TEXT,
      agent_instructions: '',
    })
  })

  it('bounds each field to its token budget', () => {
    const state = projectState([user(new TextBlock('x'.repeat(500)))], 'y'.repeat(500), {
      maxTokens: 25,
      maxInstructionTokens: 10,
    })

    expect({ request: state.request.length, instructions: state.agent_instructions.length }).toStrictEqual({
      request: 100,
      instructions: 40,
    })
    expect(state.request).toContain(OMISSION_MARKER)
  })

  it.each([
    ['maxTokens', { maxTokens: 0 }],
    ['maxInstructionTokens', { maxTokens: 1, maxInstructionTokens: -1 }],
  ])('rejects a non-positive %s', (name, options) => {
    expect(() => projectState([], undefined, options)).toThrow(`${name} must be a positive integer`)
  })
})

describe('truncateText', () => {
  it('keeps the head and tail around the marker, or hard-cuts below the marker length', () => {
    expect(truncateText('abcdefghij', 8, '..')).toBe('abc..hij')
    expect(truncateText('abcdefghij', 2, '...')).toBe('ab')
    expect(truncateText('short', 10)).toBe('short')
  })
})
