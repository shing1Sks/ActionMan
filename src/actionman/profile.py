"""Token/length profile of the prepared data -> docs/results/profile.md."""
from __future__ import annotations

import argparse
import collections
import json
import random

from .data import load_jsonl
from .features import OVERFLOW, build_pass_a, build_pass_b, gold_plan, load_tokenizer
from .schema import to_openai_tool


def pct(a, b):
    return f"{100 * a / max(b, 1):.1f}%"


def profile(tok, exs, label):
    lens_a, json_lens, buckets, arg_lens = [], [], collections.Counter(), []
    skipped_a = skipped_b = many_calls = 0
    for ex in exs:
        many_calls += len(ex.calls) > 4
        a = build_pass_a(tok, ex)
        if a is None:
            skipped_a += 1
            continue
        lens_a.append(len(a["input_ids"]))
        json_lens.append(len(tok(json.dumps([to_openai_tool(t) for t in ex.tools]) + " " + ex.query,
                                 add_special_tokens=False).input_ids))
        for (_, j, k, b) in a["len_items"]:
            buckets[b] += 1
        for c in ex.calls:
            for v in c.args.values():
                arg_lens.append(len(tok(v if isinstance(v, str) else json.dumps(v), add_special_tokens=False).input_ids))
        if ex.calls and len(ex.calls) <= 4 and build_pass_b(tok, ex, a, gold_plan(a, ex), True) is None:
            skipped_b += 1
    n = len(exs)
    s = sorted(lens_a)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))] if s else 0
    al = sorted(arg_lens)
    qa = lambda p: al[min(len(al) - 1, int(p * len(al)))] if al else 0
    names = {0: "absent", 1: "<=4", 2: "<=8", 3: "<=16", 4: "<=32", OVERFLOW: "overflow >32"}
    tot = sum(buckets.values())
    lines = [f"### {label} (n={n})", "",
             f"- Context tokens (compact signatures): p50 {q(.5)}, p90 {q(.9)}, p99 {q(.99)}, max {s[-1] if s else 0}",
             f"- Same context as OpenAI JSON schema: mean {sum(json_lens) / max(len(json_lens), 1):.0f} vs compact "
             f"{sum(lens_a) / max(len(lens_a), 1):.0f} tokens ({sum(json_lens) / max(sum(lens_a), 1):.1f}x)",
             f"- Argument value tokens: p50 {qa(.5)}, p90 {qa(.9)}, p99 {qa(.99)}, max {al[-1] if al else 0}",
             f"- Examples with > 4 calls: {many_calls} ({pct(many_calls, n)})",
             f"- Skipped: context too long {skipped_a} ({pct(skipped_a, n)}), slot block too long {skipped_b}",
             "", "| bucket | share of (call, param) items |", "|---|---|"]
    lines += [f"| {names[b]} | {pct(buckets[b], tot)} |" for b in sorted(names)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--out", default="docs/results/profile.md")
    ap.add_argument("--sample", type=int, default=8000)
    args = ap.parse_args()
    tok = load_tokenizer()
    rng = random.Random(0)
    train = load_jsonl(f"{args.data}/train.jsonl")
    bfcl = load_jsonl(f"{args.data}/bfcl.jsonl")
    out = ["# Data profile", ""]
    out.append(profile(tok, rng.sample(train, min(args.sample, len(train))), "xLAM train sample"))
    for cat in ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]:
        out.append(profile(tok, [e for e in bfcl if e.source == f"bfcl_{cat}"], f"BFCL {cat}"))
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
