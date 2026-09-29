import { context, trace } from '@opentelemetry/api'
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks'
import { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { z } from 'zod'

import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { Agent } from '../../../agent/agent.js'
import { tool } from '../../../tools/tool-factory.js'
import { FastPath, ToolCall } from '../fast-path.js'
import { MockDecisionModel, choiceAnswer } from './mock-decision-model.js'

const exporter = new InMemorySpanExporter()
const contextManager = new AsyncLocalStorageContextManager()

beforeAll(() => {
  contextManager.enable()
  context.setGlobalContextManager(contextManager)
  trace.setGlobalTracerProvider(new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] }))
})

afterAll(() => {
  trace.disable()
  context.disable()
})

const click = tool({
  name: 'click',
  description: 'Click an element',
  inputSchema: z.object({ selector: z.string() }),
  callback: ({ selector }) => `clicked ${selector}`,
})

describe('FastPath tracing', () => {
  it('records the decision span as the served cycle model span, with no chat span', async () => {
    const decisions = new MockDecisionModel([
      { action: choiceAnswer('click_submit', { click_submit: 1, other: 0 }, 0.95) },
      { action: choiceAnswer('other', { click_submit: 0.1, other: 0.9 }, 0.9) },
    ])
    const agent = new Agent({
      model: new MockMessageModel().addTurn({ type: 'textBlock', text: 'done' }),
      tools: [click],
      plugins: [
        new FastPath(decisions, { click_submit: new ToolCall('click', { selector: '#s' }) }, { minConfidence: 0.8 }),
      ],
      printer: false,
    })

    await agent.invoke('Submit the form')

    const spans = exporter.getFinishedSpans()
    const cycles = spans.filter((span) => span.name === 'execute_agent_loop_cycle')
    const decisionSpans = spans.filter((span) => span.name === 'decision')
    const chats = spans.filter((span) => span.name === 'chat')
    expect(cycles).toHaveLength(2)
    expect(decisionSpans).toHaveLength(2)
    expect(chats).toHaveLength(1)
    const [served, passed] = decisionSpans
    expect(served!.parentSpanContext?.spanId).toBe(cycles[0]!.spanContext().spanId)
    expect(served!.attributes).toMatchObject({
      'strands.source': 'decision',
      'strands.fast_path.served': true,
      'strands.fast_path.action': 'click_submit',
    })
    expect(chats[0]!.parentSpanContext?.spanId).toBe(cycles[1]!.spanContext().spanId)
    expect(passed!.parentSpanContext?.spanId).toBe(cycles[1]!.spanContext().spanId)
    expect(passed!.attributes).toMatchObject({ 'strands.fast_path.served': false, 'strands.fast_path.reason': 'other' })
    expect(passed!.attributes['strands.fast_path.action']).toBeUndefined()
  })
})
