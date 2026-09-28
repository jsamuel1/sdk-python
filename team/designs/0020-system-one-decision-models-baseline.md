# 0020 companion: System One vs LLM baseline

**Status**: Measured

**Date**: 2026-09-24; rerun 2026-09-27 with GPT-6 Luna and multi-decision tasks; self-hosted Kev-27B added 2026-09-28

**Design**: [0020 System One Decision Models](./0020-system-one-decision-models.md)

**Issue**: [#4551](https://github.com/strands-agents/harness-sdk/issues/4551)

The design asserts that a System One model is a faster and cheaper way to make typed decisions at comparable quality, and that its calibrated confidence enables a cascade no LLM classifier can drive. This document tests those claims on public labeled data before any API is built on them. Every number here comes from the scripts in [`0020-system-one-decision-models/`](./0020-system-one-decision-models/); rerun them to reproduce.

## What was measured

Six decision tasks on public splits. Three ask one question per input. Three ask several questions about the same input in one request, which is how the design expects decisions to be asked.

| Task               | Use case                    | Question(s) per input                                                                            | Data                                                                                                                                                        |
| ------------------ | --------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `banking77`        | routing / handoff           | Choice over 77 support intents                                                                   | [banking77](https://huggingface.co/datasets/legacy-datasets/banking77) test, 200 sampled                                                                    |
| `clinc_oos`        | routing with no-match       | Choice over 150 intents plus an explicit out-of-scope option                                     | [clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) `plus` test, 200 sampled (30 out-of-scope)                                                     |
| `prompt_injection` | guardrail                   | YesNo: is this a prompt injection or jailbreak?                                                  | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) test, all 116 (about one third German)                               |
| `multi_choice`     | several Choices, one input  | Choice scenario (18) **and** Choice intent (60)                                                  | [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) `en` test, 150 sampled; the two labels are joined by utterance id                     |
| `multi_yesno`      | several YesNos, one input   | YesNo anger **and** YesNo gratitude **and** YesNo curiosity                                      | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) `simplified` test, 150 sampled; gold read off the multi-label annotation |
| `mixed`            | Choice and YesNo, one input | Choice category (hate / offensive / neither) **and** YesNo offensive **and** YesNo any hate vote | [Davidson hate/offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) train, stratified 50 per class (150)                             |

The multi-task gold is read from the dataset's own annotation; none is judged by us. GoEmotions `anger` is any of anger, annoyance or disapproval, and `curiosity` is curiosity or confusion. Davidson `offensive` is "the class is not neither", and `any_hate_vote` is "at least one annotator voted hate speech", from the published per-annotator counts. A multi task scores an item **all-correct** only when every question is right, and each question is also scored on its own.

Three arms answer **identical questions**: the same instructions and the same option descriptions, with the message as state.

- **Jev** (`jev-latest`, which answered as `jev-1.13.0`) through `POST /v1/systemone`. All questions for an input go in one request.
- **GPT-6 Luna** (`global.openai.gpt-6-luna`, reasoning effort `none`, its fast mode; it rejects `temperature`) and **Claude Haiku 4.5** (temperature 0) on Amazon Bedrock (`us-west-2`), through Converse with a forced tool whose input schema is the closed answer space: an enum for Choice, a boolean for YesNo, one required field per question. A multi task is one tool call, so the LLM also answers every question for an input in one request. This is the strongest fair LLM framing: the model cannot answer outside the options and emits only a few output tokens.

The first run (2026-09-24) used **Amazon Nova Micro** as the cheap LLM. Review asked for a current fast LLM, so GPT-6 Luna replaces it here. Nova Micro was rerun with the others and is kept in [History](#history-nova-micro-and-the-first-run). A fourth arm, the open-weights **Kev-27B**, was self-hosted and run on the same items on 2026-09-28. It is reported separately in [Self-hosted: Kev-27B](#self-hosted-kev-27b), because it ran from a different host on a different day.

All arms ran on 2026-09-27 from one host in `us-west-2`, one after another, so latencies are like for like and include network time. Samples are seeded (`4551`) and one request is made per item at concurrency 4.

Prices are per million tokens:

- **Haiku 4.5** ($1.10 input / $5.50 output) and **Nova Micro** ($0.035 / $0.14): the AWS Pricing API, us-west-2 on-demand.
- **Jev** ($0.042 input, output free): [docs.typesafe.ai/models](https://docs.typesafe.ai/models), retrieved 2026-09-24.
- **GPT-6 Luna** ($0.10 / $0.50): OpenAI's short-context Standard rate. OpenAI states that Bedrock matches it. The AWS Pricing API has no Luna rows yet; retrieved 2026-09-26.

Bedrock caches long prompts automatically for Luna. On the routing tasks, 91% (banking77) and 93% (clinc_oos) of its prompt tokens are reported as `cacheWriteInputTokens`, not `inputTokens`. Every prompt token is counted here at the full input price, so Luna's cost is an upper bound.

## Results: one question per input

Accuracy with a 95% bootstrap CI (5,000 resamples). Each Δ is the paired difference on the same items, with its 95% CI. A Δ whose CI excludes zero is in bold.

| Task             | Arm           | Accuracy [95% CI]        | Δ Jev − arm [95% CI]        | p50 / p95 latency   | $ per 1k decisions |
| ---------------- | ------------- | ------------------------ | --------------------------- | ------------------- | ------------------ |
| banking77        | **Jev**       | 0.775 [0.715, 0.835]     | —                           | **0.23 s / 0.29 s** | **$0.071**         |
|                  | GPT-6 Luna    | 0.795 [0.740, 0.850]     | −0.020 [−0.065, +0.025]     | 0.63 s / 0.82 s     | $0.153             |
|                  | Haiku 4.5     | 0.760 [0.700, 0.820]     | +0.015 [−0.025, +0.055]     | 0.84 s / 0.96 s     | $2.69              |
| clinc_oos        | **Jev**       | 0.885 [0.840, 0.925]     | —                           | **0.23 s / 0.28 s** | **$0.106**         |
|                  | GPT-6 Luna    | 0.900 [0.855, 0.940]     | −0.015 [−0.060, +0.030]     | 0.63 s / 1.74 s     | $0.208             |
|                  | Haiku 4.5     | 0.880 [0.835, 0.920]     | +0.005 [−0.040, +0.050]     | 0.82 s / 0.90 s     | $3.14              |
| prompt_injection | Jev           | 0.759 [0.672, 0.836]     | —                           | 0.22 s / 0.27 s     | $0.016             |
|                  | GPT-6 Luna    | 0.836 [0.767, 0.905]     | **−0.078 [−0.138, −0.017]** | 0.61 s / 0.69 s     | $0.024             |
|                  | **Haiku 4.5** | **0.871** [0.810, 0.931] | **−0.112 [−0.172, −0.052]** | 0.75 s / 0.82 s     | $1.03              |

On prompt injection, precision was 1.00 for all three arms; recall was 0.53 for Jev, 0.68 for Luna and 0.75 for Haiku. Jev's recall was the same on English (0.54) and German (0.52) items, so its gap is not a language effect. GPT-6 Luna and Haiku 4.5 are tied with each other on all three tasks (Luna − Haiku: +0.035, +0.020 and −0.034, every CI including zero).

## Results: several questions about one input

All-correct accuracy (every question right) with a 95% bootstrap CI, and the paired Δ. Every arm answered all the questions for an input in one request.

| Task         | Arm        | All-correct [95% CI]     | Δ Jev − arm [95% CI]        | p50 / p95 latency   | $ per 1k inputs |
| ------------ | ---------- | ------------------------ | --------------------------- | ------------------- | --------------- |
| multi_choice | Jev        | 0.667 [0.587, 0.740]     | —                           | **0.22 s / 0.27 s** | **$0.063**      |
|              | GPT-6 Luna | **0.800** [0.733, 0.860] | **−0.133 [−0.207, −0.060]** | 0.64 s / 1.65 s     | $0.123          |
|              | Haiku 4.5  | 0.780 [0.713, 0.847]     | **−0.113 [−0.187, −0.040]** | 0.85 s / 0.99 s     | $2.32           |
| multi_yesno  | Jev        | 0.593 [0.513, 0.673]     | —                           | **0.22 s / 0.27 s** | **$0.014**      |
|              | GPT-6 Luna | 0.653 [0.580, 0.733]     | −0.060 [−0.127, +0.000]     | 0.66 s / 0.95 s     | $0.030          |
|              | Haiku 4.5  | 0.527 [0.447, 0.607]     | +0.067 [+0.000, +0.133]     | 0.86 s / 0.95 s     | $1.28           |
| mixed        | **Jev**    | **0.653** [0.573, 0.727] | —                           | **0.22 s / 0.30 s** | **$0.019**      |
|              | GPT-6 Luna | 0.620 [0.540, 0.700]     | +0.033 [−0.040, +0.107]     | 0.65 s / 0.77 s     | $0.040          |
|              | Haiku 4.5  | 0.613 [0.533, 0.693]     | +0.040 [−0.033, +0.113]     | 0.89 s / 1.00 s     | $1.39           |

Per question, with the paired Δ of Jev against each LLM (95% CI; bold excludes zero):

| Task         | Question      | Jev   | Luna  | Haiku | Jev − Luna                  | Jev − Haiku                 |
| ------------ | ------------- | ----- | ----- | ----- | --------------------------- | --------------------------- |
| multi_choice | scenario (18) | 0.727 | 0.867 | 0.840 | **−0.140 [−0.213, −0.060]** | **−0.113 [−0.187, −0.040]** |
|              | intent (60)   | 0.807 | 0.800 | 0.780 | +0.007 [−0.047, +0.060]     | +0.027 [−0.027, +0.080]     |
| multi_yesno  | anger         | 0.673 | 0.693 | 0.633 | −0.020 [−0.073, +0.033]     | +0.040 [−0.013, +0.100]     |
|              | gratitude     | 0.980 | 0.993 | 0.947 | −0.013 [−0.040, +0.013]     | **+0.033 [+0.007, +0.067]** |
|              | curiosity     | 0.927 | 0.947 | 0.880 | −0.020 [−0.053, +0.013]     | **+0.047 [+0.013, +0.087]** |
| mixed        | category (3)  | 0.813 | 0.713 | 0.707 | **+0.100 [+0.040, +0.160]** | **+0.107 [+0.033, +0.180]** |
|              | offensive     | 0.887 | 0.900 | 0.893 | −0.013 [−0.073, +0.047]     | −0.007 [−0.053, +0.047]     |
|              | any hate vote | 0.760 | 0.767 | 0.713 | −0.007 [−0.067, +0.060]     | +0.047 [−0.013, +0.113]     |

What the multi tasks show:

- **Jev's latency does not grow with question count.** It answered two or three questions at 0.22 s p50 and 0.27–0.30 s p95, the same as one question. Luna took 0.64–0.66 s p50 with a p95 of up to 1.65 s, and Haiku took 0.85–0.89 s.
- **Quality differs by question, not by model.** Jev's one clear loss is MASSIVE `scenario`, a coarse 18-way label: 14 points behind Luna and 11 behind Haiku. On the fine 60-way `intent` of the same utterance it ties both. Its one clear win is Davidson `category`, where it leads both LLMs by 10 to 11 points. On the three GoEmotions YesNos it ties Luna and beats Haiku on two of them.
- **Rare-positive YesNos need precision and recall, not accuracy.** GoEmotions has 7 to 13 positives per question in 150 items, so answering "no" to everything scores 0.91–0.95 on gratitude and curiosity. Per-question precision and recall are in [`results-summary.json`](./0020-system-one-decision-models/results-summary.json) under `per_question_yesno`. On `anger` (11 positives) every arm's precision is 0.16–0.17: all of them call annoyance where the annotators did not.
- **All-correct compounds.** Three questions at 0.9 each give about 0.7 all-correct. Compare a multi task's all-correct number only between arms.

### Confidence lets code trade coverage for accuracy

Jev's confidence is informative. Acting only on answers at or above a confidence floor raises accuracy on the answers kept (YesNo confidence is `|p − 0.5| × 2`):

| Task             | Floor     | Coverage  | Accuracy on covered |
| ---------------- | --------- | --------- | ------------------- |
| banking77        | 0.7 / 0.9 | 81% / 68% | 0.858 / 0.904       |
| clinc_oos        | 0.7 / 0.9 | 89% / 74% | 0.938 / 0.980       |
| prompt_injection | 0.7 / 0.9 | 78% / 64% | 0.844 / 0.905       |

The LLM arms produce no such signal: the forced tool call returns one answer with no distribution.

### The cascade the design proposes

The `DecisionAgent(fallback=...)` placement asks Jev first and sends answers below the floor to the LLM. We simulated it from the recorded per-item answers, with no new calls. Every item pays for Jev, and escalated items also pay for the LLM and its latency. The floor shown is the lowest one whose accuracy is within noise of the best.

With **Haiku 4.5** as the fallback:

| Task             | Floor | Escalated | Accuracy [95% CI]    | Δ vs Haiku-only [95% CI] | $ per 1k     | p50 / p95       |
| ---------------- | ----- | --------- | -------------------- | ------------------------ | ------------ | --------------- |
| banking77        | 0.7   | 19%       | 0.785 [0.725, 0.840] | +0.025 [+0.000, +0.050]  | $0.58 (−78%) | 0.23 s / 1.11 s |
| clinc_oos        | 0.8   | 17%       | 0.920 [0.880, 0.955] | +0.040 [+0.010, +0.070]  | $0.64 (−80%) | 0.24 s / 1.07 s |
| prompt_injection | 0.9   | 36%       | 0.862 [0.793, 0.922] | −0.009 [−0.026, +0.000]  | $0.40 (−62%) | 0.24 s / 1.04 s |

With **GPT-6 Luna** as the fallback:

| Task             | Floor | Escalated | Accuracy [95% CI]    | Δ vs Luna-only [95% CI] | $ per 1k      | p50 / p95       |
| ---------------- | ----- | --------- | -------------------- | ----------------------- | ------------- | --------------- |
| banking77        | 0.7   | 19%       | 0.800 [0.745, 0.855] | +0.005 [−0.015, +0.025] | $0.100 (−35%) | 0.23 s / 0.90 s |
| clinc_oos        | 0.8   | 17%       | 0.905 [0.860, 0.945] | +0.005 [−0.020, +0.030] | $0.141 (−32%) | 0.24 s / 0.91 s |
| prompt_injection | 0.9   | 36%       | 0.836 [0.767, 0.905] | +0.000 [+0.000, +0.000] | $0.025 (+4%)  | 0.24 s / 0.89 s |

Against Haiku, the cascade beats Haiku-only on both routing tasks at about a fifth of the cost. Against Luna it matches Luna-only on every task but does not beat it. Luna is already as accurate as Jev on routing, so escalating Jev's unsure 17–19% only recovers what Jev loses. Against a fast LLM, then, the cascade's value is cost and latency: about a third cheaper on routing, with a median of 0.23 s instead of 0.63 s. On prompt injection Luna is so cheap that a Jev first pass saves nothing.

### Run-to-run variation

Every arm was run twice on the same items: Jev, Haiku and Nova Micro on 2026-09-24 and 2026-09-27 for the single tasks, and every arm twice on 2026-09-27 otherwise. Jev gave the same answer on 195/200, 199/200 and 116/116 single-task items, and on 146–150 of 150 multi-task items; its accuracy moved by at most 0.015. Haiku and Nova Micro at temperature 0 gave the same answer on every item. GPT-6 Luna, which cannot take a temperature, gave the same answer on 197/200, 197/200 and 114/116, and on 138–145 of 150 multi items; its accuracy moved by up to 0.02. Differences of that size between two arms are noise.

### History: Nova Micro and the first run

Nova Micro (rerun 2026-09-27; same answers as 2026-09-24 on every item):

| Task             | Accuracy [95% CI]    | Δ Jev − Nova [95% CI]       | p50 / p95       | $ per 1k |
| ---------------- | -------------------- | --------------------------- | --------------- | -------- |
| banking77        | 0.610 [0.545, 0.675] | **+0.165 [+0.110, +0.220]** | 0.47 s / 0.71 s | $0.072   |
| clinc_oos        | 0.625 [0.560, 0.695] | **+0.260 [+0.190, +0.330]** | 0.46 s / 0.58 s | $0.089   |
| prompt_injection | 0.750 [0.664, 0.828] | +0.009 [−0.060, +0.078]     | 0.42 s / 0.53 s | $0.020   |

Nova Micro cost about the same as Jev and was 16 to 26 points less accurate on routing. GPT-6 Luna, which replaces it, is as accurate as Jev on routing at about twice Jev's cost.

The first run's Jev latency was 0.50–0.52 s p50 on every task. It was measured from a different host. On 2026-09-27 the same Jev requests took 0.22–0.23 s from a `us-west-2` host, with the same answers on 195–199 of 200 items, so the difference is most likely the client's network path. Jev reports no server timing that would separate the two. The first run's summaries are kept as [`results-summary-2026-09-24.json`](./0020-system-one-decision-models/results-summary-2026-09-24.json) and [`results-cascade-2026-09-24.json`](./0020-system-one-decision-models/results-cascade-2026-09-24.json).

## Self-hosted: Kev-27B

[Kev](https://github.com/jaredpalmer/kev) (Apache-2.0) is an open-weights System One model: a LoRA adapter and pointer head (adapter rev `01b81998`) on Qwen3.8-27B (rev `1d4bf0f2`), with a fitted temperature of 1.38. It serves the same `/v1/systemone` API as Jev, so `bench.py --arms kev` sends it byte-identical requests through the same TypeSafe client (`base_url=`), with no harness changes. It answers as `kev-latest`.

Setup, 2026-09-28:

- One EC2 `p5.4xlarge` (one H100 80 GB) in `us-west-2a`, running Kev's own server at kev ref `1b62aa2d`: bf16, fused kernels and CUDA graphs, one GPU, as upstream serves it.
- The client was the same `us-west-2` host as the other arms, reached through an SSM port-forward, so latency includes that tunnel. The requests, items and scoring are those of the published run, so every Δ below is paired on the same items. The other arms are the 2026-09-27 run, one day earlier.
- Cost is instance time, not tokens. The host ran about 23 minutes, and the six tasks took 152 s of wall time at concurrency 4 (966 requests). It was bought as a 7-hour EC2 Capacity Block for $39.19, because no on-demand or SageMaker capacity for a GPU that fits 27B in bf16 was available in `us-west-2`, `us-east-1` or `us-east-2` that day. A per-decision price depends on utilization, so none is given.

Accuracy with a 95% CI, and paired Δ Kev − arm (bold excludes zero):

| Task             | Kev-27B [95% CI]     | Δ vs Jev                    | Δ vs Luna                   | Δ vs Haiku                  | Kev p50 / p95   |
| ---------------- | -------------------- | --------------------------- | --------------------------- | --------------------------- | --------------- |
| banking77        | 0.815 [0.760, 0.865] | +0.040 [+0.000, +0.080]     | +0.020 [−0.025, +0.065]     | +0.055 [+0.000, +0.110]     | 0.65 s / 0.98 s |
| clinc_oos        | 0.770 [0.710, 0.825] | **−0.115 [−0.175, −0.060]** | **−0.130 [−0.190, −0.075]** | **−0.110 [−0.165, −0.060]** | 1.23 s / 1.50 s |
| prompt_injection | 0.759 [0.681, 0.836] | +0.000 [−0.060, +0.060]     | **−0.078 [−0.129, −0.034]** | **−0.112 [−0.172, −0.060]** | 0.46 s / 0.64 s |
| multi_choice     | 0.640 [0.560, 0.713] | −0.027 [−0.087, +0.033]     | **−0.160 [−0.227, −0.093]** | **−0.140 [−0.207, −0.073]** | 0.60 s / 0.92 s |
| multi_yesno      | 0.673 [0.600, 0.753] | **+0.080 [+0.020, +0.140]** | +0.020 [−0.040, +0.080]     | **+0.147 [+0.080, +0.220]** | 0.22 s / 0.49 s |
| mixed            | 0.547 [0.467, 0.627] | **−0.107 [−0.180, −0.033]** | −0.073 [−0.153, +0.007]     | −0.067 [−0.133, +0.000]     | 0.32 s / 0.57 s |

What Kev-27B shows:

- **It matches Jev on routing with a small label set and on injection, and loses on a large one.** It ties or edges Jev on banking77 (77 intents). It trails Jev and both LLMs by 11 to 13 points on clinc_oos (150 intents plus out-of-scope). On prompt injection its accuracy, precision (1.00) and recall (0.53, the same on English and German) equal Jev's, so it shares Jev's recall gap to the LLMs.
- **Per question it differs from Jev in both directions.** It is 7 points better on GoEmotions `anger` ([+0.007, +0.120] vs Jev) and 7 points worse on Davidson `category` ([−0.133, −0.013]). On `anger` it says yes less often: precision 0.18 and recall 0.73, against Jev's 0.16 and 0.82. On a question with 11 positives in 150 items, fewer false alarms is most of an accuracy gain. Per-question numbers are in `results-summary.json` under `kev`.
- **Its latency grows with the request, Jev's does not.** On one H100 Kev's p50 was 0.22 s on the three short GoEmotions YesNos and 0.32 s on `mixed`, 0.46 s on prompt injection (one YesNo over longer texts), 0.60–0.65 s on the 77- and 60-way Choices, and 1.23 s on clinc_oos (151 options). Jev stayed at 0.22–0.23 s on every task. Kev runs each question as its own row continuing from the state, with a Choice's options inside that row, so a long option list is a long row. The client-side token counts do not explain the difference: Jev reports more tokens than Kev on clinc_oos (about 2,530 in against 1,140).
- **Its confidence is informative and more conservative than Jev's.** At a 0.7 floor Kev covers 69% of items on every single task, against Jev's 78–89%, and is right on 0.94, 0.91 and 0.84 of what it covers. At 0.9 it covers 36–54% and is right on 1.00, 1.00 and 0.90. The same floor therefore escalates more on Kev, which matches the samples: tune floors per engine.

As a cascade front, Kev escalated 31% at a 0.7 floor and 38–44% at 0.8 on the single tasks. Against Haiku it beats Haiku-only on banking77 (floor 0.8: +0.020 [+0.005, +0.040], 40% escalated) and reaches parity on the other two at 0.8–0.9. Against Luna it reaches Luna-only's accuracy at 0.8–0.9 (38–64% escalated) and does not beat it. Per floor: [`results-cascade-kev.json`](./0020-system-one-decision-models/results-cascade-kev.json) (Haiku fallback) and [`results-cascade-kev-luna.json`](./0020-system-one-decision-models/results-cascade-kev-luna.json) (Luna fallback). Their `$ per 1k` is the escalated LLM spend only: Kev's own cost is the instance time above.

For Strands, Kev-27B confirms the design's premise that `DecisionModel` is not one vendor. The same schema, client and adapters run against an open-weights model on your own hardware, with calibrated confidence. It is not a drop-in replacement for Jev on every task, so measure it on your own questions like any other engine.

## What this supports, and what it does not

**Supported.**

- **Latency.** Jev answers at 0.22–0.23 s p50 and under 0.31 s p95, whether it is asked one question or three. That is about 0.4 s faster than GPT-6 Luna and 0.6 s faster than Haiku 4.5 at the median. Luna's p95 reaches 1.7 s.
- **Cost.**
  - Against GPT-6 Luna, Jev costs about half as much per decision on routing ($0.071 vs $0.153 on banking77; $0.106 vs $0.208 on clinc_oos) and on the multi tasks. Luna's figure is an upper bound (see the pricing note), so the ratio could be smaller if cached prompt tokens are discounted.
  - Against Haiku 4.5, Jev is 30–38× cheaper on routing and 37–89× cheaper on the multi tasks.
- **Quality on routing.** Jev is statistically tied with both GPT-6 Luna and Haiku on both routing tasks: every paired CI includes zero.
- **Calibrated confidence is the differentiator.** The confidence lets code decide when to pay for an LLM, and that claim holds.
  - Against Haiku, the cascade beats Haiku-only on routing accuracy, with CIs excluding zero on clinc_oos and touching zero on banking77, at about a fifth of the cost.
  - Against GPT-6 Luna, it matches accuracy at lower cost and lower latency.
  - The claim that the cascade _improves_ accuracy holds against Haiku but not against Luna.

**Not supported, or not yet.**

- **Guardrails are not a clean win.** On prompt injection, Jev alone trails Luna by 8 points and Haiku by 11, all of it recall. The cascade closes the gap to within noise, but only with a high floor (0.9) that escalates 36% of traffic, and against Luna it saves nothing. The design already defaults guards to fail closed and shows the cascade. The docs must not claim System One is as accurate as an LLM for security classification. It is a cheap, fast first pass that knows when it is unsure.
- **Not every question in a multi-decision schema is a tie.** Jev trails both LLMs by 11 to 14 points on MASSIVE's coarse `scenario` label while tying on the fine `intent`, and leads both on Davidson `category`. Check a multi-decision schema per question, not per model.
- **Wording was not tuned per arm.** One question wording served all arms. A prompt tuned for one model could move its numbers, so treat each arm's result as a lower bound.
- **Small samples.** With 116 to 200 items per task, the CIs are ±4 to 8 points. The ties are "not distinguishable at this n", not proof of equality.
- **Not measured here:**
  - Model selection quality: samples 2 and 3 in the P0 plan cover it.
  - Non-English routing.

## Consequences for the design

1. Keep the cascade (`fallback=` with a confidence floor) as the recommended placement for anything security-adjacent, and say so in the placement guide. Against a fast LLM, present it as a cost and latency control, not an accuracy gain.
2. `DecisionGuard` keeps failing closed by default. Its docs should cite this measurement rather than claim parity.
3. Asking several questions in one request stays the default: it is where System One's latency advantage is largest, and the multi tasks show no systematic quality cost from asking together.
4. The samples' `--engine llm` switch should stay. The right engine is an empirical question per task, and here per question, and letting one schema run on both engines is how to answer it.

## Reproduce

```bash
cd team/designs/0020-system-one-decision-models
python fetch_datasets.py                       # public slices -> data/
export TYPESAFE_API_KEY=...; export AWS_PROFILE=...   # Bedrock in us-west-2
python bench.py --tasks all --arms jev,global.openai.gpt-6-luna,us.anthropic.claude-haiku-4-5-20251001-v1:0
python bench.py --tasks single --arms us.amazon.nova-micro-v1:0
python analyze.py --tasks all --arms jev=jev,luna=global.openai.gpt-6-luna,haiku=us.anthropic.claude-haiku-4-5-20251001-v1:0
python analyze.py                              # Jev vs Haiku and Nova Micro, the first run's arms
python cascade.py                              # Haiku fallback
python cascade.py --slow global.openai.gpt-6-luna --out cascade-luna.json
KEV_BASE_URL=http://127.0.0.1:8010 python bench.py --tasks all --arms kev   # any /v1/systemone Kev server
python analyze.py --tasks all --arms kev=kev,jev=jev,luna=global.openai.gpt-6-luna,haiku=us.anthropic.claude-haiku-4-5-20251001-v1:0
python cascade.py --fast kev --out cascade-kev.json
python cascade.py --fast kev --slow global.openai.gpt-6-luna --out cascade-kev-luna.json
LOGIT_BASE_URL=http://127.0.0.1:8010/v1/logits LOGIT_TEMPERATURE=1.7 \
  python bench.py --tasks all --arms logit     # any generic HTTP logit server
```

The `logit` arm puts a local logit model in the same results table as Jev, Kev and the LLMs. It needs no `/v1/systemone` server. For each question it POSTs `{"instruction", "text", "labels"}` (YesNo sends `["true", "false"]`) and reads back `{"logits": {label: float}}`, plus an optional `"model"`. Probabilities are `softmax(logits / LOGIT_TEMPERATURE)`. A raw-logit model is uncalibrated until a temperature is fitted (see `fit_temperature`), so the arm reports confidence, and a selective-accuracy curve, only when `LOGIT_TEMPERATURE` is set. It sends one request per question, so its latency is the slowest question in a multi-question task, not one call. No logit model has been run through it yet; a contributed logit provider is the first planned arm.

`analyze.py --arms short=id,...` and `cascade.py --fast <id> --slow <id>` take the same arm ids as `bench.py --arms`, so any other arm runs through the same analysis. Accuracy counts an errored item (an API or parse failure) as incorrect. Latency, tokens and cost are computed over non-errored items. Every arm's `errors` count is in the summary. In the published runs it is 0 for every arm and task, so no reported accuracy includes an error.

Committed summaries:

- [`results-summary.json`](./0020-system-one-decision-models/results-summary.json): every arm and task.
- [`results-cascade.json`](./0020-system-one-decision-models/results-cascade.json): the Haiku fallback.
- [`results-cascade-luna.json`](./0020-system-one-decision-models/results-cascade-luna.json): the Luna fallback. Its keys keep the published names `haiku_only` and `vs_haiku` for the fallback arm.
- [`results-cascade-kev.json`](./0020-system-one-decision-models/results-cascade-kev.json) and [`results-cascade-kev-luna.json`](./0020-system-one-decision-models/results-cascade-kev-luna.json): Kev-27B as the front, with Haiku and Luna fallbacks (same key names).

Per-item results are regenerated by `bench.py`.
