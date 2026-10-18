# Model routing example

Runs `ModelRouter` with `ClassifierStrategy` and `DecisionStrategy` against real Bedrock models. Each scenario prints which model served the first call and whether that matches the expected candidate.

| Scenario | What it checks |
|----------|----------------|
| 1. `ClassifierStrategy` | The existing constructor and options still work. A routine request is served by Nova Lite and a design request by Nova Pro. |
| 2. `DecisionStrategy` + `LLMDecisionModel` | Routing works with a general-purpose LLM as the decision model. |
| 3. `DecisionStrategy` + custom `DecisionModel` | A local keyword model routes without making a classifier call. The script prints the question (positional keys `c0`, `c1`) and the bounded JSON state the decision model receives. |
| 4. Timeout | A decision model that never answers is aborted through `cancelSignal`, and the router serves candidate `c0`. |

## Prerequisites

- Node.js 22+
- AWS credentials with Bedrock access to `us.anthropic.claude-haiku-4-5-20251001-v1:0`, `us.amazon.nova-lite-v1:0`, and `us.amazon.nova-pro-v1:0`

## Run

```bash
npm install
npm start
```

The SDK is linked from `../../`, so local SDK changes are picked up after `npm run build` in `strands-ts/`. The script exits non-zero when any check fails. Scenarios 1 and 2 depend on a live classifier, so an occasional mis-route there is model variance, not an SDK failure.
