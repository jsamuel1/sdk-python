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

// --8<-- [start:fast_path_imports]
import { Agent } from '@strands-agents/sdk'
import { FastPath, ToolCall } from '@strands-agents/sdk/experimental'
import { TypeSafeDecisionModel } from '@strands-agents/sdk/models/typesafe'
// --8<-- [end:fast_path_imports]
