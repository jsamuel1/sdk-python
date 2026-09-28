"""Fetch small, public, labeled evaluation slices via the HuggingFace datasets-server (no auth).

Writes JSONL under data/. Deterministic: seeded sample over a fixed split.
Usage: python fetch_datasets.py            # fetch all task slices
       python fetch_datasets.py --info DS   # print splits/features for discovery
"""

import json
import pathlib
import random
import sys
import urllib.parse
import urllib.request

OUT = pathlib.Path(__file__).parent / "data"
API = "https://datasets-server.huggingface.co"
SEED = 4551

# task -> (dataset, config, split, label_field, sample_size or None for all)
SLICES = {
    "banking77": ("legacy-datasets/banking77", "default", "test", "label", 200),
    "clinc_oos": ("clinc/clinc_oos", "plus", "test", "intent", 200),
    "prompt_injection": ("deepset/prompt-injections", "default", "test", "label", None),
}

# Multi-decision tasks: several questions about ONE input, asked together. Each item's gold is a dict keyed by
# question id.
MULTI_SAMPLE = 150
MASSIVE = ("mteb/amazon_massive_intent", "mteb/amazon_massive_scenario", "en", "test")
GOEMOTIONS = ("google-research-datasets/go_emotions", "simplified", "test")
ANGER = {"anger", "annoyance", "disapproval"}
# Davidson et al. 2017 (MIT): 3-way class plus per-annotator counts; ~6% hate speech, so stratified to 50 per class.
HATE_OFFENSIVE = ("tdavidson/hate_speech_offensive", "default", "train", 50)


def get(path, **params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def label_names(dataset, config, field):
    feats = get("info", dataset=dataset)["dataset_info"][config]["features"]
    spec = feats[field]
    return spec.get("names") if isinstance(spec, dict) else None


def fetch_split(dataset, config, split):
    """Download the split's parquet export once (avoids per-row rate limits)."""
    import io

    import pyarrow.parquet as pq

    files = [
        f for f in get("parquet", dataset=dataset)["parquet_files"] if f["config"] == config and f["split"] == split
    ]
    rows = []
    for f in files:
        with urllib.request.urlopen(f["url"], timeout=120) as r:
            rows += pq.read_table(io.BytesIO(r.read())).to_pylist()
    return rows


def fetch_massive(rng):
    """2 Choices per utterance: scenario (18) and intent (60). Same rows, joined by id."""
    intent_ds, scenario_ds, cfg, split = MASSIVE
    intents = {r["id"]: r for r in fetch_split(intent_ds, cfg, split)}
    scenarios = {r["id"]: r["label_text"] for r in fetch_split(scenario_ds, cfg, split)}
    ids = sorted(i for i in intents if i in scenarios)
    rows = [intents[i] for i in sorted(rng.sample(ids, MULTI_SAMPLE))]
    labels = {
        "scenario": sorted(set(scenarios.values())),
        "intent": sorted({r["label_text"] for r in intents.values()}),
    }
    items = [{"text": r["text"], "gold": {"scenario": scenarios[r["id"]], "intent": r["label_text"]}} for r in rows]
    return items, labels


def goemotions_gold(names):
    """Gold for 3 YesNo read directly off the multi-label annotation (no derived labels)."""
    s = set(names)
    return {"anger": bool(s & ANGER), "gratitude": "gratitude" in s, "curiosity": bool(s & {"curiosity", "confusion"})}


def fetch_goemotions(rng):
    ds, cfg, split = GOEMOTIONS
    names = get("info", dataset=ds)["dataset_info"][cfg]["features"]["labels"]["feature"]["names"]
    rows = fetch_split(ds, cfg, split)
    rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), MULTI_SAMPLE))]
    items = [{"text": r["text"], "gold": goemotions_gold([names[k] for k in r["labels"]])} for r in rows]
    return items, {}


def fetch_hate_offensive(rng):
    """1 Choice (the dataset's own 3-way class) + 2 YesNo read off the same annotation.

    ``offensive``: the class is not "neither" (a coarsening of the 3-way label). ``any_hate_vote``: at least one
    annotator voted hate speech (the published per-annotator count). Neither is a new judgment.
    """
    ds, cfg, split, per_class = HATE_OFFENSIVE
    names = label_names(ds, cfg, "class")
    rows = fetch_split(ds, cfg, split)
    by_class = {k: [i for i, r in enumerate(rows) if r["class"] == k] for k in range(len(names))}
    picked = sorted(i for k in by_class for i in rng.sample(by_class[k], per_class))
    items = []
    for i in picked:
        r = rows[i]
        gold = {
            "category": names[r["class"]],
            "offensive": names[r["class"]] != "neither",
            "any_hate_vote": r["hate_speech_count"] > 0,
        }
        items.append({"text": r["tweet"], "gold": gold})
    return items, {"category": names}


def write(task, items, labels):
    with open(OUT / f"{task}.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    (OUT / f"{task}.labels.json").write_text(json.dumps(labels))
    print(task, len(items), "rows")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for task, (ds, cfg, split, field, n) in SLICES.items():
        meta = get("info", dataset=ds)["dataset_info"][cfg]
        total = meta["splits"][split]["num_examples"]
        names = label_names(ds, cfg, field)
        all_rows = fetch_split(ds, cfg, split)
        assert len(all_rows) == total, (task, len(all_rows), total)
        rows = all_rows if n is None else [all_rows[i] for i in sorted(rng.sample(range(total), n))]
        with open(OUT / f"{task}.jsonl", "w") as f:
            for r in rows:
                gold = names[r[field]] if names else r[field]
                f.write(json.dumps({"text": r["text"], "gold": gold}) + "\n")
        (OUT / f"{task}.labels.json").write_text(json.dumps(names if names else sorted({0, 1})))
        print(task, len(rows), "rows;", len(names) if names else 2, "labels;", f"{ds}/{cfg}/{split}")
    # Separate generator per multi task so adding one never reshuffles the others (or the single tasks above).
    write("massive_multi", *fetch_massive(random.Random(SEED + 1)))
    write("goemotions_multi", *fetch_goemotions(random.Random(SEED + 2)))
    write("hate_offensive_multi", *fetch_hate_offensive(random.Random(SEED + 3)))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--info"]:
        for ds in sys.argv[2:]:
            for cfg, c in get("info", dataset=ds)["dataset_info"].items():
                print(
                    ds, cfg, {s: v["num_examples"] for s, v in c.get("splits", {}).items()}, list(c.get("features", {}))
                )
    else:
        main()
