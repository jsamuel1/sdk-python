import { z } from 'zod'
import { Agent, BedrockModel, ModelRouter, RoutingCandidate, tool } from '@strands-agents/sdk'
import { DecisionStrategy, FastPath, ToolCall, choice, score, yesNo } from '@strands-agents/sdk/experimental'
import { TypeSafeDecisionModel } from '@strands-agents/sdk/models/typesafe'

async function firstQuestion(): Promise<void> {
  // --8<-- [start:first_question]
  const Triage = z.object({
    department: choice({
      billing: 'Charges and refunds',
      technical: 'Bugs and outages',
      account: null,
    }).describe('Which team should handle `ticket`?'),
    urgent: yesNo('Does `ticket` convey time pressure?'),
    frustration: score(['calm', 'upset', 'furious'], 'How frustrated is the customer?'),
  })

  const jev = new TypeSafeDecisionModel()
  const ticket = { ticket: 'You billed me twice. Fix it today!' }
  const decision = await jev.decide(Triage, ticket)
  console.log(decision.output)
  console.log(decision.answers.department!.confidence)
  // --8<-- [end:first_question]
}

function modelSelection(): Agent {
  // --8<-- [start:model_selection]
  const router = new ModelRouter(
    [
      new RoutingCandidate({
        model: new BedrockModel({ modelId: 'us.anthropic.claude-sonnet-4-6' }),
        name: 'complex',
        description: 'Multi-step reasoning, proofs, code generation',
      }),
      new RoutingCandidate({
        model: new BedrockModel({ modelId: 'us.amazon.nova-micro-v1:0' }),
        name: 'routine',
        description: 'Direct questions, short summaries, extraction',
      }),
    ],
    {
      strategy: new DecisionStrategy(new TypeSafeDecisionModel(), {
        minConfidence: 0.7,
      }),
    }
  )
  const agent = new Agent({ model: router })
  // --8<-- [end:model_selection]
  return agent
}

const click = tool({
  name: 'click',
  description: 'Click an element',
  inputSchema: z.object({ selector: z.string() }),
  callback: ({ selector }) => `clicked ${selector}`,
})
const scroll = tool({
  name: 'scroll',
  description: 'Scroll the page',
  inputSchema: z.object({ dy: z.number() }),
  callback: ({ dy }) => `scrolled ${dy}`,
})

function fastPath(): Agent {
  // --8<-- [start:fast_path]
  const fastPath = new FastPath(
    new TypeSafeDecisionModel(),
    {
      click_submit: new ToolCall('click', { selector: '#submit' }),
      scroll_down: new ToolCall('scroll', { dy: 600 }, 'Scroll one screen down'),
    },
    { minConfidence: 0.8 }
  )
  const agent = new Agent({ tools: [click, scroll], plugins: [fastPath] })
  // --8<-- [end:fast_path]
  return agent
}

void firstQuestion
void modelSelection
void fastPath
