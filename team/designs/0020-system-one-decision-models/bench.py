"""System One (Jev) vs LLM baseline on identical typed decision questions.

Arms:
  jev          TypeSafe /v1/systemone, jev-latest
  kev          A self-hosted Kev server at KEV_BASE_URL (same /v1/systemone API), kev-latest
  logit        Any generic HTTP logit server at LOGIT_BASE_URL: POST {instruction, text, labels} and get
               {"logits": {label: float}} back, one request per question. Probabilities are
               softmax(logits / LOGIT_TEMPERATURE); confidence is reported only when LOGIT_TEMPERATURE is set
               (a raw-logit model is uncalibrated until a temperature is fitted; see fit_temperature).
  <bedrock id> Bedrock Converse with a forced tool whose schema is the same closed answer space
               (enum for Choice, boolean for YesNo), temperature 0 (gpt-6-luna: reasoning effort none; it rejects
               temperature). A multi-question task gets one tool field per question, in one call.

Tasks (public labeled data, fetched by fetch_datasets.py; seed 4551). One question per input:
  banking77          Choice over 77 intents                         (routing / handoff)
  clinc_oos          Choice over 150 intents + "oos" no-match        (routing with explicit no-match)
  prompt_injection   YesNo "is this a prompt injection / jailbreak"  (guardrail)
Several questions about one input, asked together in one request:
  multi_choice       MASSIVE (en): Choice scenario (18) + Choice intent (60)
  multi_yesno        GoEmotions: YesNo anger + YesNo gratitude + YesNo curiosity
  mixed              Davidson hate/offensive: Choice category (3, stratified) + YesNo offensive + YesNo any-hate-vote

Per arm: accuracy, precision/recall (YesNo), p50/p95 latency, input/output tokens (Bedrock prompt-cache reads and
writes included, at the full input price), $/1k requests (null for a self-hosted arm), wall time per task,
and for Jev and Kev a selective-accuracy curve (accuracy vs coverage at confidence thresholds). Multi tasks report
all-correct accuracy (every question right) plus per-question accuracy.
Accuracy counts an errored item (API or parse failure) as incorrect; the error count is reported per arm as
`errors`, and latency, tokens and cost are over non-errored items only.

Usage: bench.py --arms jev,kev,global.openai.gpt-6-luna --tasks all|single|multi|<t1,t2> [--limit N] [--concurrency 4]
Writes results/<task>__<arm>.jsonl (one row per item, raw) and results/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import pathlib
import statistics
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

# USD per 1M tokens (input, output). Sources: AWS Pricing API (us-west-2 on-demand, geo/US profile) and
# docs.typesafe.ai/models (Jev: input only, output free). Retrieved 2026-09-24.
# gpt-6-luna has no AWS Pricing API rows yet; its price is OpenAI's short-context Standard rate, which OpenAI states
# Bedrock matches in commercial regions (developers.openai.com/api/docs/pricing, retrieved 2026-09-26). A self-hosted
# Kev (or a generic logit server) has no per-token price: its cost is instance time, so it reports NaN.
PRICES = {
    "jev": (0.042, 0.0),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.10, 5.50),
    "us.amazon.nova-micro-v1:0": (0.035, 0.14),
    "global.openai.gpt-6-luna": (0.10, 0.50),
}
# Per-model Converse settings. gpt-6-luna rejects `temperature` and reasons by default; reasoning effort "none" is
# its fast, non-reasoning mode (measured: 0.6 s vs 0.9-2.2 s with default reasoning on a two-field forced tool).
BEDROCK_OPTIONS = {
    "global.openai.gpt-6-luna": {
        "inferenceConfig": {"maxTokens": 512},
        "additionalModelRequestFields": {"reasoning": {"effort": "none"}},
    },
}
DEFAULT_BEDROCK_OPTIONS = {"inferenceConfig": {"temperature": 0, "maxTokens": 256}}
SINGLE_TASKS = ["banking77", "clinc_oos", "prompt_injection"]
MULTI_TASKS = ["multi_choice", "multi_yesno", "mixed"]

# --------------------------------------------------------------------------- questions


def humanize(label: str) -> str:
    return label.replace("_", " ")


GOEMOTIONS_YESNO = {
    "anger": {
        "type": "noul",
        "instructions": "Does the writer express anger, annoyance, or disapproval?",
    },
    "gratitude": {"type": "noul", "instructions": "Does the writer express gratitude or thanks?"},
    "curiosity": {
        "type": "noul",
        "instructions": "Does the writer express curiosity or confusion, such as wanting to know or understand more?",
    },
}
HATE_OFFENSIVE = {
    "category": {
        "type": "choice",
        "instructions": "How would you classify this tweet?",
        "criteria": {
            "hate speech": "Expresses hatred toward a group, or is intended to be derogatory, humiliating or insulting "
            "to members of a group, based on race, religion, gender, sexuality, or similar",
            "offensive language": "Offensive or vulgar, but not hate speech",
            "neither": "Neither hate speech nor offensive",
        },
    },
    "offensive": {
        "type": "noul",
        "instructions": "Is this tweet hate speech or offensive language?",
    },
    "any_hate_vote": {
        "type": "noul",
        "instructions": "Could a reasonable reader consider this tweet hate speech toward a group?",
    },
}


def multi_spec(task: str) -> tuple[str, dict]:
    """Return (data file, questions) for a multi-decision task: several questions about one input, asked together."""
    if task == "multi_choice":
        labels = json.loads((DATA / "massive_multi.labels.json").read_text())
        return "massive_multi", {
            "scenario": {
                "type": "choice",
                "instructions": "Which area of a voice assistant does the user request belong to?",
                "criteria": {lab: humanize(lab) for lab in labels["scenario"]},
            },
            "intent": {
                "type": "choice",
                "instructions": "Which specific voice assistant intent does the user request express?",
                "criteria": {lab: humanize(lab) for lab in labels["intent"]},
            },
        }
    if task == "multi_yesno":
        return "goemotions_multi", dict(GOEMOTIONS_YESNO)
    if task == "mixed":
        return "hate_offensive_multi", dict(HATE_OFFENSIVE)
    raise ValueError(task)


def questions_for(task: str) -> tuple[str, dict, bool]:
    """Return (data file, questions keyed by id, multi?). Single tasks ask one question under the id ``q``."""
    if task in MULTI_TASKS:
        data, questions = multi_spec(task)
        return data, questions, True
    q, _ = task_spec(task)
    return task, {"q": q}, False


def task_spec(task: str):
    """Return (question dict in Jev wire format, gold normaliser) for a single-question task."""
    labels = json.loads((DATA / f"{task}.labels.json").read_text())
    if task == "banking77":
        q = {
            "type": "choice",
            "instructions": "Which banking support intent does the customer message express?",
            "criteria": {lab: humanize(lab) for lab in labels},
        }
        return q, (lambda g: g)
    if task == "clinc_oos":
        crit = {lab: humanize(lab) for lab in labels if lab != "oos"}
        crit["oos"] = "Out of scope: the request matches none of the other intents"
        q = {
            "type": "choice",
            "instructions": "Which assistant intent does the user utterance express? Choose oos if none apply.",
            "criteria": crit,
        }
        return q, (lambda g: g)
    if task == "prompt_injection":
        q = {
            "type": "noul",
            "instructions": "Is this text a prompt injection or jailbreak attempt: does it try to override, ignore, or "
            "replace an AI assistant's instructions, or make it adopt a different role or behavior?",
            "criteria": {
                "true": "Attempts to override or replace the assistant's instructions or role",
                "false": "An ordinary request or statement, with no attempt to override instructions",
            },
        }
        return q, (lambda g: bool(g))
    raise ValueError(task)


# --------------------------------------------------------------------------- arms


def answer_value(q: dict, a: dict):
    return a["choice"] if q["type"] == "choice" else a["noul"] >= 0.5


def answer_confidence(q: dict, a: dict):
    return a.get("confidence") if q["type"] == "choice" else abs(a["noul"] - 0.5) * 2


class SystemOneArm:
    """Any /v1/systemone server: TypeSafe's hosted Jev (default) or a self-hosted Kev (``kev``, at KEV_BASE_URL)."""

    def __init__(self, name: str = "jev"):
        self.name = name
        if name == "kev":
            self.url = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8009").rstrip("/") + "/v1/systemone"
            self.model = "kev-latest"
            self._key = os.environ.get("KEV_API_KEY") or "local"  # a TypeSafe key is never sent to Kev
        else:
            self.url = "https://api.typesafe.ai/v1/systemone"
            self.model = "jev-latest"
            self._key = os.environ["TYPESAFE_API_KEY"]

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 529) and attempt < 5:
                    time.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError("unreachable")

    async def ask(self, text: str, questions: dict) -> dict:
        body = {"state": {"message": text}, "model": self.model, "questions": questions}
        t = time.perf_counter()
        out = await asyncio.to_thread(self._post, body)
        lat = time.perf_counter() - t
        answers = out["answers"]
        return {
            "preds": {qid: answer_value(q, answers[qid]) for qid, q in questions.items()},
            "confs": {qid: answer_confidence(q, answers[qid]) for qid, q in questions.items()},
            "raw": answers,
            "lat": lat,
            "in": out["usage"]["input_tokens"] or 0,
            "out": out["usage"]["output_tokens"] or 0,
            "model": out["model"],
        }


class HttpLogitArm:
    """A generic logit server: instruction, text and labels in, one logit per label out.

    The request body is ``{"instruction": str, "text": str, "labels": [str, ...]}`` (YesNo sends
    ``["true", "false"]``); the response is ``{"logits": {label: float}}`` and may add ``"model"``. The endpoint is
    ``LOGIT_BASE_URL`` (default ``http://127.0.0.1:8010/v1/logits``) with an optional ``LOGIT_API_KEY`` bearer token.
    """

    name = "logit"

    def __init__(self):
        self.url = os.environ.get("LOGIT_BASE_URL", "http://127.0.0.1:8010/v1/logits")
        self._key = os.environ.get("LOGIT_API_KEY")
        raw_t = os.environ.get("LOGIT_TEMPERATURE")
        self.temperature = float(raw_t) if raw_t else None
        if self.temperature is not None and not (self.temperature > 0 and math.isfinite(self.temperature)):
            raise ValueError(f"LOGIT_TEMPERATURE must be a finite number above 0, got {raw_t!r}")

    @staticmethod
    def _labels(q: dict) -> list[str]:
        return list(q["criteria"]) if q["type"] == "choice" else ["true", "false"]

    @staticmethod
    def _instruction(q: dict) -> str:
        # The same content Jev and the LLM arms receive: instructions plus option descriptions.
        return BedrockArm._question_text(q)

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)

    def answer(self, q: dict, logits: dict) -> tuple[object, float | None, dict]:
        """Map one question's logits to (prediction, confidence, probabilities) at the configured temperature."""
        labels = self._labels(q)
        if set(logits) != set(labels):
            raise ValueError(f"logit server returned labels {sorted(logits)}, expected {sorted(labels)}")
        t = self.temperature or 1.0
        top = max(logits.values())
        weights = {lab: math.exp((v - top) / t) for lab, v in logits.items()}
        total = sum(weights.values())
        probs = {lab: w / total for lab, w in weights.items()}
        if q["type"] == "choice":
            pred = max(probs, key=probs.__getitem__)
            conf = probs[pred]
        else:
            pred = probs["true"] >= 0.5
            conf = abs(2 * probs["true"] - 1)
        return pred, (conf if self.temperature is not None else None), probs

    async def ask(self, text: str, questions: dict) -> dict:
        bodies = {
            qid: {"instruction": self._instruction(q), "text": text, "labels": self._labels(q)}
            for qid, q in questions.items()
        }
        t = time.perf_counter()
        outs = await asyncio.gather(*(asyncio.to_thread(self._post, body) for body in bodies.values()))
        lat = time.perf_counter() - t
        by_q = dict(zip(questions, outs, strict=True))
        mapped = {qid: self.answer(q, by_q[qid]["logits"]) for qid, q in questions.items()}
        return {
            "preds": {qid: m[0] for qid, m in mapped.items()},
            "confs": {qid: m[1] for qid, m in mapped.items()},
            "raw": {qid: {"logits": by_q[qid]["logits"], "probabilities": mapped[qid][2]} for qid in questions},
            "lat": lat,
            "in": 0,
            "out": 0,
            "model": next((o.get("model") for o in outs if o.get("model")), self.url),
        }


class BedrockArm:
    def __init__(self, model_id: str, profile: str, region: str):
        import boto3
        from botocore.config import Config

        self.name = model_id
        self._c = boto3.Session(profile_name=profile, region_name=region).client(
            "bedrock-runtime", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=120)
        )

    @staticmethod
    def _field(q: dict) -> dict:
        if q["type"] == "choice":
            return {"type": "string", "enum": list(q["criteria"]), "description": "The selected option."}
        return {"type": "boolean", "description": "True if the condition holds."}

    @classmethod
    def _tool(cls, questions: dict) -> dict:
        # One field per question, so every question is answered in one forced tool call (the LLM's "ask together").
        # A single question keeps its historical field name, "answer".
        names = {qid: ("answer" if list(questions) == ["q"] else qid) for qid in questions}
        props = {names[qid]: cls._field(q) for qid, q in questions.items()}
        schema = {"type": "object", "properties": props, "required": list(props)}
        return {"toolSpec": {"name": "answer", "description": "Record the answer.", "inputSchema": {"json": schema}}}

    @staticmethod
    def _question_text(q: dict) -> str:
        # Same content Jev receives: instructions + option descriptions, nothing more.
        if q["type"] == "choice":
            opts = "\n".join(f"- {k}: {v}" for k, v in q["criteria"].items())
            return f"{q['instructions']}\n\nOptions:\n{opts}"
        c = q.get("criteria", {})
        return f"{q['instructions']}\n\ntrue: {c.get('true', 'yes')}\nfalse: {c.get('false', 'no')}"

    @classmethod
    def _prompt(cls, text: str, questions: dict) -> str:
        if list(questions) == ["q"]:
            body = cls._question_text(questions["q"])
        else:
            body = "\n\n".join(f"## {qid}\n{cls._question_text(q)}" for qid, q in questions.items())
        return f"{body}\n\n<message>\n{text}\n</message>\n\nAnswer by calling the answer tool."

    def _call(self, text: str, questions: dict) -> dict:
        return self._c.converse(
            modelId=self.name,
            messages=[{"role": "user", "content": [{"text": self._prompt(text, questions)}]}],
            toolConfig={"tools": [self._tool(questions)], "toolChoice": {"tool": {"name": "answer"}}},
            **BEDROCK_OPTIONS.get(self.name, DEFAULT_BEDROCK_OPTIONS),
        )

    async def ask(self, text: str, questions: dict) -> dict:
        t = time.perf_counter()
        r = await asyncio.to_thread(self._call, text, questions)
        lat = time.perf_counter() - t
        got = {}
        for block in r["output"]["message"]["content"]:
            if "toolUse" in block:
                got = block["toolUse"]["input"]
        single = list(questions) == ["q"]
        usage = r["usage"]
        # Bedrock caches long prompts automatically for some models (gpt-6-luna reports ~93% of a routing prompt as
        # cacheWriteInputTokens). Every prompt token is counted and priced at the full input rate, so a cached arm's
        # cost is an upper bound rather than silently low.
        cached = usage.get("cacheReadInputTokens", 0) + usage.get("cacheWriteInputTokens", 0)
        return {
            "preds": {qid: got.get("answer" if single else qid) for qid in questions},
            "confs": {qid: None for qid in questions},
            "raw": got,
            "lat": lat,
            "in": usage["inputTokens"] + cached,
            "cached_in": cached,
            "out": r["usage"]["outputTokens"],
            "model": self.name,
        }


# --------------------------------------------------------------------------- run + score


def score_row(r: dict, gold, multi: bool) -> dict:
    """Single tasks keep the flat pred/conf/gold shape; multi tasks add per-question correctness and all-correct."""
    preds, confs = r.pop("preds"), r.pop("confs")
    if not multi:
        r.update({"pred": preds.get("q"), "conf": confs.get("q"), "gold": gold})
        r["correct"] = r["pred"] == gold
        return r
    per_q = {qid: preds.get(qid) == gold[qid] for qid in preds}
    r.update({"preds": preds, "confs": confs, "gold": gold, "per_q": per_q, "correct": all(per_q.values())})
    return r


async def run_arm(arm, task: str, items: list[dict], concurrency: int) -> list[dict]:
    _, questions, multi = questions_for(task)
    norm = (lambda g: g) if multi else task_spec(task)[1]
    sem = asyncio.Semaphore(concurrency)

    async def one(i, item):
        async with sem:
            try:
                r = await arm.ask(item["text"], questions)
                r["error"] = None
            except Exception as e:  # noqa: BLE001 - record and continue; errors are reported, not hidden
                empty = {qid: None for qid in questions}
                r = {"preds": empty, "confs": dict(empty), "raw": None, "lat": None, "in": 0, "out": 0}
                r["error"] = repr(e)[:300]
            r["i"] = i
            return score_row(r, norm(item["gold"]), multi)

    return await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def _precision_recall(preds: list, golds: list[bool]) -> dict:
    tp = sum(p is True and g for p, g in zip(preds, golds, strict=True))
    fp = sum(p is True and not g for p, g in zip(preds, golds, strict=True))
    fn = sum(p is not True and g for p, g in zip(preds, golds, strict=True))
    return {
        "positives": sum(golds),
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, tp + fn), 4),
    }


def summarize(arm: str, task: str, rows: list[dict], wall_s: float = float("nan")) -> dict:
    ok = [r for r in rows if r["error"] is None]
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    lat = [r["lat"] for r in ok]
    tin = sum(r["in"] for r in ok)
    tout = sum(r["out"] for r in ok)
    pin, pout = PRICES.get(arm, (float("nan"), float("nan")))
    cost_per_1k = (tin * pin + tout * pout) / 1e6 / max(1, len(ok)) * 1000
    s = {
        "arm": arm,
        "task": task,
        "n": n,
        "errors": n - len(ok),
        "accuracy": round(acc, 4),
        "p50_s": round(statistics.median(lat), 3) if lat else None,
        "p95_s": round(pct(lat, 0.95), 3) if lat else None,
        "mean_in_tokens": round(tin / max(1, len(ok)), 1),
        "mean_out_tokens": round(tout / max(1, len(ok)), 1),
        # A self-hosted arm has no per-token price (its cost is instance time): null, so summary.json stays valid JSON.
        "usd_per_1k": None if cost_per_1k != cost_per_1k else round(cost_per_1k, 5),
        "wall_s": round(wall_s, 1),
        "answered_by": sorted({r.get("model") for r in ok if r.get("model")}),
    }
    if "per_q" in rows[0]:
        s["questions"] = len(rows[0]["per_q"])
        s["per_question_accuracy"] = {qid: round(sum(r["per_q"][qid] for r in rows) / n, 4) for qid in rows[0]["per_q"]}
        # YesNo positives are rare in these slices, so accuracy alone flatters "always no": report P/R as well.
        for qid in rows[0]["per_q"]:
            if isinstance(rows[0]["gold"][qid], bool):
                s.setdefault("per_question_yesno", {})[qid] = _precision_recall(
                    [r["preds"][qid] for r in rows], [r["gold"][qid] for r in rows]
                )
        return s
    if isinstance(rows[0]["gold"], bool):
        tp = sum(r["pred"] is True and r["gold"] for r in rows)
        fp = sum(r["pred"] is True and not r["gold"] for r in rows)
        fn = sum(r["pred"] is not True and r["gold"] for r in rows)
        s["precision"] = round(tp / max(1, tp + fp), 4)
        s["recall"] = round(tp / max(1, tp + fn), 4)
    if any(r["conf"] is not None for r in ok):
        curve = []
        for th in (0.0, 0.3, 0.5, 0.7, 0.9):
            kept = [r for r in ok if r["conf"] is not None and r["conf"] >= th]
            if kept:
                curve.append(
                    {
                        "min_conf": th,
                        "coverage": round(len(kept) / n, 3),
                        "accuracy": round(sum(r["correct"] for r in kept) / len(kept), 4),
                    }
                )
        s["selective"] = curve
    return s


def make_arm(name: str, profile: str, region: str):
    if name in ("jev", "kev"):
        return SystemOneArm(name)
    if name == "logit":
        return HttpLogitArm()
    return BedrockArm(name, profile, region)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="jev")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--profile", default=None, help="AWS profile for the Bedrock arms (default: environment)")
    ap.add_argument("--region", default="us-west-2")
    a = ap.parse_args()
    named = {"all": SINGLE_TASKS + MULTI_TASKS, "single": SINGLE_TASKS, "multi": MULTI_TASKS}
    tasks = [t for name in a.tasks.split(",") for t in named.get(name, [name])]
    RESULTS.mkdir(exist_ok=True)
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for arm_name in a.arms.split(","):
        arm = make_arm(arm_name, a.profile, a.region)
        for task in tasks:
            data_file = questions_for(task)[0]
            items = [json.loads(line) for line in open(DATA / f"{data_file}.jsonl")][: a.limit]
            started = time.perf_counter()
            rows = await run_arm(arm, task, items, a.concurrency)
            wall_s = time.perf_counter() - started
            slug = arm_name.replace("/", "_").replace(":", "_")
            with open(RESULTS / f"{task}__{slug}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
            s = summarize(arm_name, task, rows, wall_s)
            summary[f"{task}::{arm_name}"] = s
            print(json.dumps(s))
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
