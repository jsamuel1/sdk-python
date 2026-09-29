import { DecisionModel } from '../decision-model.js'
import type { AskOptions, DecisionModelConfig } from '../decision-model.js'
import { ChoiceAnswer, DecisionResponse, ScoreAnswer, YesNoAnswer, yesNoConfidence } from '../types.js'
import type { Answer, DecisionState, Question } from '../types.js'

type Scripted = Readonly<Record<string, Answer>> | Error

/**
 * A scripted decision model that records every request. Pass one answer map (or error) for every call, or an array
 * to answer successive calls in order; an exhausted queue repeats its last entry.
 */
export class MockDecisionModel extends DecisionModel {
  readonly requests: Array<{ state: DecisionState; questions: Readonly<Record<string, Question>> }> = []
  private _config: DecisionModelConfig = { modelId: 'mock-s1' }
  private readonly _queue: Scripted[]

  constructor(
    answers: Scripted | readonly Scripted[] = {},
    private readonly _calibrated = true
  ) {
    super()
    this._queue = Array.isArray(answers) ? [...answers] : [answers as Scripted]
  }

  override get calibrated(): boolean {
    return this._calibrated
  }

  getConfig(): DecisionModelConfig {
    return this._config
  }

  updateConfig(config: DecisionModelConfig): void {
    this._config = { ...this._config, ...config }
  }

  protected async _ask(
    state: DecisionState,
    questions: Readonly<Record<string, Question>>,
    _options: AskOptions
  ): Promise<DecisionResponse> {
    this.requests.push({ state, questions })
    const answers = this._queue.length > 1 ? this._queue.shift()! : this._queue[0]!
    if (answers instanceof Error) throw answers
    return new DecisionResponse(answers, 'mock-s1-1.0', { inputTokens: 10, outputTokens: 2, totalTokens: 12 })
  }
}

export function choiceAnswer(
  choice: string,
  probabilities: Readonly<Record<string, number>>,
  confidence?: number
): ChoiceAnswer {
  return new ChoiceAnswer(choice, probabilities, confidence)
}

export function yes(probability: number): YesNoAnswer {
  return new YesNoAnswer(probability, yesNoConfidence(probability))
}

export function scoreAnswer(value: number, probabilities: Readonly<Record<number, number>>): ScoreAnswer {
  return new ScoreAnswer(value, probabilities, 0.5)
}
