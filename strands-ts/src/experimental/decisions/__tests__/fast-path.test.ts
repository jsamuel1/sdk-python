import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { z } from 'zod'

import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { MockSnapshotStorage } from '../../../__fixtures__/mock-storage-provider.js'
import { Agent } from '../../../agent/agent.js'
import { AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent } from '../../../hooks/events.js'
import { logger } from '../../../logging/logger.js'
import { SessionManager } from '../../../session/session-manager.js'
import { synthesizedSource } from '../../../agent/synthesized.js'
import { tool } from '../../../tools/tool-factory.js'
import { ImageBlock } from '../../../types/media.js'
import { JsonBlock, Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import { FastPath, ToolCall, latestToolResultText } from '../fast-path.js'
import { Choice, DecisionResponse } from '../types.js'
import type { ChoiceAnswer } from '../types.js'
import { MockDecisionModel, choiceAnswer } from './mock-decision-model.js'
import type { DecisionModel } from '../decision-model.js'

const clicks: string[] = []

const click = tool({
  name: 'click',
  description: 'Click an element',
  inputSchema: z.object({ selector: z.string() }),
  callback: ({ selector }) => {
    clicks.push(selector)
    return `clicked ${selector}`
  },
})

const scroll = tool({
  name: 'scroll',
  description: 'Scroll the page',
  inputSchema: z.object({ dy: z.number() }),
  callback: ({ dy }) => `scrolled ${dy}`,
})

const ACTIONS = {
  click_submit: new ToolCall('click', { selector: '#submit' }),
  scroll_down: new ToolCall('scroll', { dy: 600 }, 'Scroll one screen down'),
}

function served(option = 'click_submit', confidence: number | null = 0.95): Record<string, ChoiceAnswer> {
  const probabilities = { click_submit: 0, scroll_down: 0, other: 0, [option]: 1 }
  return { action: choiceAnswer(option, probabilities, confidence ?? undefined) }
}

const OTHER = { action: choiceAnswer('other', { click_submit: 0.1, scroll_down: 0.1, other: 0.8 }, 0.8) }

function llm(...texts: string[]): MockMessageModel {
  const model = new MockMessageModel()
  for (const text of texts) model.addTurn({ type: 'textBlock', text })
  return model
}

function agent(decisions: DecisionModel, model: MockMessageModel, options: Partial<{ maxConsecutive: number }> = {}) {
  return new Agent({
    model,
    tools: [click, scroll],
    plugins: [new FastPath(decisions, ACTIONS, { minConfidence: 0.8, ...options })],
    printer: false,
  })
}

beforeEach(() => {
  clicks.length = 0
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('FastPath', () => {
  it('serves a confident step with a marked synthesized turn, then the LLM resumes', async () => {
    const model = llm('submitted')
    const stream = vi.spyOn(model, 'stream')
    const target = agent(new MockDecisionModel([served(), OTHER]), model)

    const result = await target.invoke('Submit the form')

    expect(result.toString().trim()).toBe('submitted')
    expect(clicks).toStrictEqual(['#submit'])
    expect(stream).toHaveBeenCalledOnce()
    const synthesized = target.messages[1]!
    const toolUse = synthesized.content[0]!
    expect(toolUse.type).toBe('toolUseBlock')
    if (toolUse.type !== 'toolUseBlock') return
    expect(toolUse.name).toBe('click')
    expect(toolUse.input).toStrictEqual({ selector: '#submit' })
    expect(toolUse.toolUseId).toMatch(/^s1-/)
    expect(synthesized.metadata).toStrictEqual({
      custom: { strands: { source: 'decision', decision_model: 'mock-s1-1.0', confidence: 0.95 } },
    })
    const toolResult = target.messages[2]!.content[0]
    expect(toolResult).toBeInstanceOf(ToolResultBlock)
    expect((toolResult as ToolResultBlock).toolUseId).toBe(toolUse.toolUseId)
  })

  it.each([
    ['the configured model id', 'mock-s1'],
    ['null', undefined],
  ])('attributes to %s when the response names no model', async (_label, modelId) => {
    class Anonymous extends MockDecisionModel {
      protected override async _ask(...args: Parameters<MockDecisionModel['_ask']>) {
        const response = await super._ask(...args)
        return new DecisionResponse(response.answers, undefined, response.usage)
      }
    }
    const decisions = new Anonymous([served(), OTHER])
    decisions.updateConfig(modelId === undefined ? {} : { modelId })
    if (modelId === undefined) delete (decisions.getConfig() as { modelId?: string }).modelId
    const target = agent(decisions, llm('done'))

    await target.invoke('Submit the form')

    expect((target.messages[1]!.metadata?.custom?.strands as { decision_model: unknown }).decision_model).toBe(
      modelId ?? null
    )
  })

  it('asks every action plus other and projects the latest tool result', async () => {
    const decisions = new MockDecisionModel([served(), OTHER])
    await agent(decisions, llm('done')).invoke('Submit the form')

    const first = decisions.requests[0]!
    expect(first.state).toStrictEqual({ request: 'Submit the form', agent_instructions: '' })
    expect((first.questions.action as Choice).options).toStrictEqual({
      click_submit: '{"tool":"click","input":{"selector":"#submit"}}',
      scroll_down: 'Scroll one screen down',
      other: 'None of the actions fits; let the reasoning model decide',
    })
    expect((decisions.requests[1]!.state as Record<string, string>).latest_tool_result).toBe('clicked #submit')
  })

  it.each([
    ['other', OTHER],
    ['below floor', served('click_submit', 0.79)],
    ['no confidence', served('click_submit', null)],
    ['decision error', new Error('service down')],
  ])('passes %s through to the LLM unchanged', async (_label, answer) => {
    const model = llm('llm answer')
    const target = agent(new MockDecisionModel([answer]), model)

    const result = await target.invoke('Submit the form')

    expect(result.toString().trim()).toBe('llm answer')
    expect(clicks).toStrictEqual([])
    expect(target.messages[1]!.metadata?.custom).toBeUndefined()
  })

  it('logs a decision error by type only', async () => {
    const warn = vi.spyOn(logger, 'warn')
    await agent(new MockDecisionModel([new Error('secret url')]), llm('llm answer')).invoke('Submit the form')

    expect(warn).toHaveBeenCalledWith(expect.stringContaining('reason=<decision_error>, error_type=<Error>'))
    expect(warn.mock.calls.flat().join(' ')).not.toContain('secret url')
  })

  it('passes through when the chosen action tool is not offered', async () => {
    const target = new Agent({
      model: llm('llm answer'),
      tools: [scroll],
      plugins: [new FastPath(new MockDecisionModel([served()]), ACTIONS, { minConfidence: 0.8 })],
      printer: false,
    })

    expect((await target.invoke('Submit the form')).toString().trim()).toBe('llm answer')
  })

  it('forces an LLM turn after maxConsecutive synthesized turns', async () => {
    const decisions = new MockDecisionModel([served(), served(), served()])
    const model = llm('llm turn')
    const stream = vi.spyOn(model, 'stream')

    await agent(decisions, model, { maxConsecutive: 2 }).invoke('Keep submitting')

    expect(clicks).toStrictEqual(['#submit', '#submit'])
    expect(decisions.requests).toHaveLength(2)
    expect(stream).toHaveBeenCalledOnce()
  })

  it('resets the consecutive count after a model turn', async () => {
    const decisions = new MockDecisionModel([served(), served()])
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'scroll', toolUseId: 't1', input: { dy: 1 } })
      .addTurn({ type: 'textBlock', text: 'done' })

    await agent(decisions, model, { maxConsecutive: 1 }).invoke('Submit, scroll, submit')

    // served, forced model turn, served again after the model turn, forced again
    expect(clicks).toStrictEqual(['#submit', '#submit'])
    expect(decisions.requests).toHaveLength(2)
  })

  it('skips the fast path when a tool choice is forced', async () => {
    const decisions = new MockDecisionModel([OTHER, served()])
    const model = new MockMessageModel()
      .addTurn({ type: 'textBlock', text: 'no tool' })
      .addTurn({ type: 'toolUseBlock', name: 'strands_structured_output', toolUseId: 'so', input: { text: 'ok' } })
    const target = agent(decisions, model)

    const result = await target.invoke('Answer', { structuredOutputSchema: z.object({ text: z.string() }) })

    expect(result.structuredOutput).toStrictEqual({ text: 'ok' })
    expect(decisions.requests).toHaveLength(1)
  })

  it('fires no AfterModelCallEvent for a synthesized turn while tool hooks run', async () => {
    const target = agent(new MockDecisionModel([served(), OTHER]), llm('done'))
    const before: BeforeModelCallEvent[] = []
    const after: AfterModelCallEvent[] = []
    const tools: string[] = []
    target.addHook(BeforeModelCallEvent, (event) => void before.push(event))
    target.addHook(AfterModelCallEvent, (event) => void after.push(event))
    target.addHook(AfterToolCallEvent, (event) => void tools.push(event.toolUse.name))

    await target.invoke('Submit the form')

    expect(before).toHaveLength(2)
    expect(after.map((event) => (event.stopData?.message.content[0] as TextBlock).text)).toStrictEqual(['done'])
    expect(tools).toStrictEqual(['click'])
  })

  it('bills only decision usage', async () => {
    const model = new MockMessageModel().addTurn(
      { type: 'textBlock', text: 'done' },
      { usage: { inputTokens: 100, outputTokens: 5, totalTokens: 105 } }
    )
    const target = agent(new MockDecisionModel([served(), served(), OTHER]), model)

    const result = await target.invoke('Submit twice')

    expect(result.metrics!.accumulatedUsage.totalTokens).toBe(105)
    expect(target.messages[1]!.metadata?.usage).toBeUndefined()
  })

  it('round-trips the marking through a session', async () => {
    const storage = new MockSnapshotStorage()
    const session = (): SessionManager => new SessionManager({ sessionId: 's', storage: { snapshot: storage } })
    const first = new Agent({
      id: 'browser',
      model: llm('done'),
      tools: [click, scroll],
      plugins: [new FastPath(new MockDecisionModel([served(), OTHER]), ACTIONS, { minConfidence: 0.8 })],
      sessionManager: session(),
      printer: false,
    })
    await first.invoke('Submit the form')
    const toolUseId = (first.messages[1]!.content[0] as { toolUseId: string }).toolUseId

    const restored = new Agent({ id: 'browser', model: llm('unused'), sessionManager: session(), printer: false })
    await restored.initialize()

    const synthesized = restored.messages[1]!
    expect((synthesized.content[0] as { toolUseId: string }).toolUseId).toBe(toolUseId)
    expect(toolUseId).toMatch(/^s1-/)
    expect(synthesized.metadata?.custom).toStrictEqual({
      strands: { source: 'decision', decision_model: 'mock-s1-1.0', confidence: 0.95 },
    })
    expect(restored.messages[3]!.metadata?.custom).toBeUndefined()
  })

  describe('constructor', () => {
    it('refuses an uncalibrated model', () => {
      expect(() => new FastPath(new MockDecisionModel({}, false), ACTIONS, { minConfidence: 0.8 })).toThrow(
        'FastPath(minConfidence=0.8) needs a calibrated DecisionModel'
      )
    })

    it('refuses a non-DecisionModel', () => {
      expect(() => new FastPath({} as DecisionModel, ACTIONS, { minConfidence: 0.8 })).toThrow(TypeError)
    })

    it.each([
      [{}, {}, 'at least one action'],
      [{ other: new ToolCall('click') }, {}, "'other' is reserved"],
      [{ a: { name: 'click', input: {} } as unknown as ToolCall }, {}, 'must be a ToolCall'],
      [ACTIONS, { minConfidence: 0 }, 'minConfidence'],
      [ACTIONS, { minConfidence: 1.5 }, 'minConfidence'],
      [ACTIONS, { maxConsecutive: 0 }, 'maxConsecutive'],
      [ACTIONS, { maxRequestTokens: 1.5 }, 'maxRequestTokens'],
    ])('refuses invalid configuration %#', (actions, options, message) => {
      expect(
        () =>
          new FastPath(new MockDecisionModel(), actions as Record<string, ToolCall>, { minConfidence: 0.8, ...options })
      ).toThrow(message)
    })
  })

  describe('synthesizedSource', () => {
    it.each([
      [undefined, undefined],
      [{ custom: { strands: 'decision' } }, undefined],
      [{ custom: { strands: ['decision'] } }, undefined],
      [{ custom: { strands: null } }, undefined],
      [{ custom: { strands: { source: 7 } } }, undefined],
      [{ custom: { strands: { source: 'decision' } } }, 'decision'],
    ])('reads %j as %s', (metadata, expected) => {
      const message = new Message({ role: 'assistant', content: [], ...(metadata && { metadata }) })
      expect(synthesizedSource(message)).toBe(expected)
    })
  })

  describe('latestToolResultText', () => {
    it('labels errors and media and bounds the text', () => {
      const result = new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 't0',
            status: 'error',
            content: [
              new JsonBlock({ json: { code: 404 } }),
              new ImageBlock({ format: 'png', source: { bytes: new Uint8Array() } }),
              new TextBlock('x'.repeat(50)),
            ],
          }),
          new TextBlock('ignored'),
        ],
      })

      expect(latestToolResultText([result], 1_000)).toBe(`[error]\n{"code":404}\n[Image]\n${'x'.repeat(50)}`)
      expect(latestToolResultText([result], 40)).toHaveLength(40)
      expect(latestToolResultText([new Message({ role: 'user', content: [new TextBlock('hi')] })], 100)).toBeUndefined()
      expect(latestToolResultText([], 100)).toBeUndefined()
    })
  })
})
