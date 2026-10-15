"""Build data/prepared/{train,dev,test,bfcl}.jsonl from data/raw. Run once on the laptop."""
from __future__ import annotations

import argparse
import json
import os
import random
import re

from .data import irrelevance_negatives, save_jsonl
from .schema import from_bfcl, from_xlam

BFCL_CATS = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]


def load_bfcl(raw: str):
    out = []
    for cat in BFCL_CATS:
        qs = [json.loads(l) for l in open(os.path.join(raw, "bfcl", f"BFCL_v3_{cat}.json"), encoding="utf-8")]
        ans = {}
        ap = os.path.join(raw, "bfcl", "possible_answer", f"BFCL_v3_{cat}.json")
        if os.path.exists(ap):
            ans = {r["id"]: r for r in (json.loads(l) for l in open(ap, encoding="utf-8"))}
        out += [from_bfcl(q, ans.get(q["id"]), cat) for q in qs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--negatives", type=int, default=5000)
    args = ap.parse_args()
    rng = random.Random(0)
    rows = json.load(open(os.path.join(args.raw, "xlam_function_calling_60k.json"), encoding="utf-8"))
    exs = [from_xlam(r) for r in rows]
    rng.shuffle(exs)
    test, dev, train = exs[:1000], exs[1000:2000], exs[2000:]
    train += irrelevance_negatives(train, args.negatives, rng)
    dev += irrelevance_negatives(dev, 100, rng)
    test += irrelevance_negatives(test, 100, rng)
    rng.shuffle(train)
    os.makedirs(args.out, exist_ok=True)
    for name, part in [("train", train), ("dev", dev), ("test", test), ("bfcl", load_bfcl(args.raw))]:
        save_jsonl(os.path.join(args.out, f"{name}.jsonl"), part)
        print(name, len(part))



def prepare_plus(raw: str = "data/raw", out: str = "data/prepared"):
    """Stage 5 'more data': xLAM train + ToolACE + Hermes single-turn -> train_plus.jsonl."""
    from .data import load_jsonl
    from .extra import from_hermes, from_toolace
    rng = random.Random(1)
    base = load_jsonl(os.path.join(out, "train.jsonl"))
    herm = [e for e in (from_hermes(r) for r in json.load(open(os.path.join(raw, "extra", "hermes_singleturn.json"), encoding="utf-8"))) if e]
    tace = [e for e in (from_toolace(r, i) for i, r in enumerate(json.load(open(os.path.join(raw, "extra", "toolace.json"), encoding="utf-8")))) if e]
    allx = base + herm + tace
    rng.shuffle(allx)
    save_jsonl(os.path.join(out, "train_plus.jsonl"), allx)
    print("train_plus", len(allx), {"xlam+neg": len(base), "hermes": len(herm), "toolace": len(tace)})


def prepare_v2(out: str = "data/prepared", n_hard: int = 6000):
    """v2 train set: train_plus + hard negatives (needed tool removed, related tools kept)."""
    from .data import hard_negatives, load_jsonl
    rng = random.Random(2)
    plus = load_jsonl(os.path.join(out, "train_plus.jsonl"))
    xlam = [e for e in plus if e.source == "xlam"]
    hard = hard_negatives(xlam, n_hard, rng)
    allx = plus + hard
    rng.shuffle(allx)
    save_jsonl(os.path.join(out, "train_v2.jsonl"), allx)
    print("train_v2", len(allx), "hard negatives", len(hard))


V4_CAPS = {"synth": 45000, "glaive_call": 25000, "glaive_nocall": 4000, "toucan": 15000}


def _qkey(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def prepare_v4(raw: str = "data/raw/extra2", out: str = "data/prepared", caps: dict = V4_CAPS):
    """v4 train set: train_v2 + Synth-APIGen + Glaive v2 (first turn) + Toucan (first step).
    Drops any example whose query matches a dev/test/BFCL query, mostly-non-English queries, and duplicate queries."""
    import pandas as pd
    from .data import load_jsonl
    from .extra import from_glaive, from_synth_apigen, from_toucan
    rng = random.Random(4)
    seen = {_qkey(e.query) for n in ("dev", "test", "bfcl", "train_v2") for e in load_jsonl(os.path.join(out, f"{n}.jsonl"))}
    held = {_qkey(e.query) for n in ("dev", "test", "bfcl") for e in load_jsonl(os.path.join(out, f"{n}.jsonl"))}
    f = lambda name: os.path.join(raw, name)
    pools = {
        "synth": [from_synth_apigen(r, i) for i, r in
                  enumerate(pd.read_parquet(f("Synth-APIGen-v0.1__train-00000-of-00001.parquet")).to_dict("records"))],
        "glaive": [from_glaive(r, i) for i, r in
                   enumerate(json.load(open(f("glaive-function-calling-v2__glaive-function-calling-v2.json"), encoding="utf-8")))],
        "toucan": [from_toucan(r, i) for i, r in
                   enumerate(pd.read_parquet(f("Toucan-1.5M__train-00000-of-00003.parquet")).to_dict("records"))],
    }
    stats, new = {}, []
    for name, exs in pools.items():
        exs = [e for e in exs if e is not None]
        n_conv = len(exs)
        keep = []
        for e in exs:
            k = _qkey(e.query)
            if k in held:
                stats.setdefault("dropped_eval_overlap", 0)
                stats["dropped_eval_overlap"] += 1
                continue
            if k in seen or sum(ch.isascii() for ch in e.query) < 0.97 * len(e.query):
                continue
            seen.add(k)
            keep.append(e)
        rng.shuffle(keep)
        if name == "glaive":
            calls = [e for e in keep if e.calls][: caps["glaive_call"]]
            keep = calls + [e for e in keep if not e.calls][: caps["glaive_nocall"]]
        else:
            keep = keep[: caps[name]]
        stats[name] = {"converted": n_conv, "kept": len(keep), "no_call": sum(not e.calls for e in keep)}
        new += keep
    allx = load_jsonl(os.path.join(out, "train_v2.jsonl")) + new
    rng.shuffle(allx)
    save_jsonl(os.path.join(out, "train_v4.jsonl"), allx)
    print("train_v4", len(allx), json.dumps(stats))
    return stats


if __name__ == "__main__":
    import sys
    if "--v4" in sys.argv:
        prepare_v4()
    elif "--plus" in sys.argv:
        prepare_plus()
    elif "--v2" in sys.argv:
        prepare_v2()
    else:
        main()
