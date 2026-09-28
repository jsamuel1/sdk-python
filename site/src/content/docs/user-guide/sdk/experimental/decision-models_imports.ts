// @ts-nocheck
// --8<-- [start:first_question_imports]
import { z } from 'zod'
import { choice, score, yesNo } from '@strands-agents/sdk/experimental'
import { TypeSafeDecisionModel } from '@strands-agents/sdk/models/typesafe'
// --8<-- [end:first_question_imports]

// --8<-- [start:model_selection_imports]
import { Agent, BedrockModel, ModelRouter, RoutingCandidate } from '@strands-agents/sdk'
import { DecisionStrategy } from '@strands-agents/sdk/experimental'
import { TypeSafeDecisionModel } from '@strands-agents/sdk/models/typesafe'
// --8<-- [end:model_selection_imports]

// --8<-- [start:swarm_handoff_imports]
import { Agent, Swarm } from '@strands-agents/sdk'
import { DecisionHandoffStrategy } from '@strands-agents/sdk/experimental'
import { TypeSafeDecisionModel } from '@strands-agents/sdk/models/typesafe'
// --8<-- [end:swarm_handoff_imports]
