import { z } from 'zod'
import {
  Agent,
  BedrockModel,
  ModelRouter,
  RoutingCandidate,
  Swarm,
} from '@strands-agents/sdk'
import {
  DecisionHandoffStrategy,
  DecisionStrategy,
  choice,
  score,
  yesNo,
} from '@strands-agents/sdk/experimental'
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

async function swarmHandoff(): Promise<void> {
  // --8<-- [start:swarm_handoff]
  const swarm = new Swarm({
    nodes: [
      new Agent({ id: 'triage', systemPrompt: "You summarise the customer's problem." }),
      new Agent({ id: 'billing', description: 'Charges, invoices, refunds' }),
      new Agent({ id: 'technical', description: 'Bugs, outages, errors' }),
    ],
    handoffStrategy: new DecisionHandoffStrategy(new TypeSafeDecisionModel(), {
      minConfidence: 0.7,
    }),
  })
  const result = await swarm.invoke('The export button throws a 500')
  // --8<-- [end:swarm_handoff]
  void result
}

void firstQuestion
void modelSelection
void swarmHandoff
