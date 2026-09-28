/**
 * System One decision models: typed, calibrated decisions as a first-class primitive.
 *
 * A `Model` generates; a `DecisionModel` decides. Declare questions as a Zod schema (or `Choice` / `Score` /
 * `YesNo` objects), ask them together about some state, and get typed answers with probabilities.
 *
 * Experimental: subject to change without notice. See `team/designs/0020-system-one-decision-models.md`.
 */
export { DecisionModel } from './decision-model.js'
export { answerFromLogits, logitsOf, softmax } from './calibration.js'
export type { AnswerFromLogitsOptions } from './calibration.js'
export type { AskOptions, DecisionModelConfig } from './decision-model.js'
export { DecisionStrategy } from './decision-strategy.js'
export { COMPLETE_OPTION, DECISION_STATE_KEY, DecisionHandoffStrategy } from './decision-handoff-strategy.js'
export type { DecisionHandoffStrategyOptions, HandoffDecision } from './decision-handoff-strategy.js'
export type { DecisionStrategyOptions } from './decision-strategy.js'
export { CompiledSchema, NO_MATCH_OPTION, choice, compileSchema, score, yesNo } from './schema.js'
export type { DecisionSchema, ScoreField, YesNoFieldOptions } from './schema.js'
export {
  Choice,
  ChoiceAnswer,
  DecisionResponse,
  Score,
  ScoreAnswer,
  YesNo,
  YesNoAnswer,
  maxProbabilityConfidence,
  yesNoConfidence,
} from './types.js'
export type {
  Answer,
  Decision,
  DecisionState,
  JSONContent,
  Question,
  RawScores,
  YesNoKey,
  YesNoOptions,
} from './types.js'
export { projectState } from '../../models/request-text.js'
export type { ProjectedState, ProjectStateOptions } from '../../models/request-text.js'
