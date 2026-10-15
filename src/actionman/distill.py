"""v3 distillation: a strong LLM re-writes every training call; we keep its surface form, never its facts.

On the box:  python -m actionman.distill --model Qwen/Qwen3-32B-FP8 --in data/prepared/train_v2.jsonl --out $OUT/d
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from collections import Counter

from . import beat
from .schema import Call, gold_from_calls, value_to_text
from .scoring import _std, value_match


def _word_in(text: str, hay: str) -> bool:
    text = text.strip()
    return bool(text) and re.search(r"(?<!\w)" + re.escape(text.lower()) + r"(?!\w)", hay.lower()) is not None


def _merge_value(o, t, query: str):
    """Teacher value only if it states the same fact as the original:
    scorer-equal; or a longer string the user literally wrote (word-aligned in the query) that contains the original;
    or a shorter string made of whole words of the original. Never between differing digit strings."""
    if isinstance(o, list) and isinstance(t, list) and len(o) == len(t):
        return [_merge_value(a, b, query) for a, b in zip(o, t)]
    if type(o) is not type(t):
        return o
    if value_match(t, o):
        return t
    if isinstance(o, str):
        no, nt = _std(o), _std(t)
        if not (no and nt) or (no.isdigit() and nt.isdigit()):
            return o
        if no in nt and _word_in(t, query):
            return t
        if nt in no and _word_in(t, o) and not nt.isdigit():
            return t
    return o


def merge_teacher(orig: list[Call], teacher: list[Call] | None, query: str,
                  params: dict[str, set] | None = None) -> tuple[list[Call], Counter]:
    """params: declared parameter names per tool; teacher-added args must be declared and word-aligned in the query."""
    st = Counter(calls=len(orig))
    if teacher is None or sorted(c.name for c in teacher) != sorted(c.name for c in orig):
        st["rejected_call"] = len(orig)
        return copy.deepcopy(orig), st
    pool: dict[str, list[Call]] = {}
    for c in teacher:
        pool.setdefault(c.name, []).append(c)
    out = []
    for c in orig:
        t = pool[c.name].pop(0)               # pair same-name calls in order
        args = {}
        for k, v in c.args.items():
            st["args"] += 1
            nv = _merge_value(v, t.args[k], query) if k in t.args else v
            if nv != v:
                st["changed"] += 1
            args[k] = nv
        declared = (params or {}).get(c.name)
        for k, v in t.args.items():
            if k in c.args or (declared is not None and k not in declared):
                continue
            if isinstance(v, (str, int, float)) and not isinstance(v, bool) and _word_in(value_to_text(v), query):
                args[k] = v
                st["added"] += 1
        out.append(Call(c.name, args))
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B-FP8")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--pilot", type=int, default=0, help="run this many first; continue only if the gates pass")
    ap.add_argument("--gate-changed", type=float, default=3.0, help="min %% of arg values the teacher changes")
    ap.add_argument("--gate-hours", type=float, default=1.75, help="max projected hours for the full run")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baseline"))
    from qwen import parse_calls, prompt_text   # same prompt + parser as the Qwen baseline
    from vllm import LLM, SamplingParams
    from .data import load_jsonl, save_jsonl

    os.makedirs(args.out, exist_ok=True)
    exs = load_jsonl(args.inp)
    todo = [i for i, e in enumerate(exs) if e.calls]
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    llm = LLM(args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.92, enable_prefix_caching=True)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=384)
    total, n_tok_in, n_tok_out, skipped, gated = Counter(), 0, 0, 0, False
    t1 = time.time()
    for s in range(0, len(todo), args.chunk):
        idx = todo[s: s + args.chunk]
        prompts, keep = [], []
        for i in idx:
            p = prompt_text(tok, exs[i])
            if len(tok(p, add_special_tokens=False).input_ids) > args.max_model_len - 400:
                skipped += 1
                continue
            prompts.append(p)
            keep.append(i)
        for i, res in zip(keep, llm.generate(prompts, sp, use_tqdm=False)):
            text = res.outputs[0].text
            n_tok_in += len(res.prompt_token_ids)
            n_tok_out += len(res.outputs[0].token_ids)
            teacher = parse_calls(text) or None
            params = {t.name: {p.name for p in t.params} for t in exs[i].tools}
            calls, st = merge_teacher(exs[i].calls, teacher, exs[i].query, params)
            total.update(st)
            exs[i].calls = calls
            exs[i].gold = gold_from_calls(exs[i].tools, calls)
        beat()
        el = time.time() - t1
        done = min(s + args.chunk, len(todo))
        print(f"{done}/{len(todo)} {el:.0f}s changed={total['changed']}/{total['args']} "
              f"rejected={total['rejected_call']}/{total['calls']}", flush=True)
        if args.pilot and not gated and done >= min(args.pilot, len(todo)) and done < len(todo):
            gated = True
            per = (time.time() - t1) / done
            proj_h = per * (len(todo) - done) / 3600
            pct = 100 * total["changed"] / max(total["args"], 1)
            gate = {"pilot_examples": done, "changed_pct": pct, "projected_rest_h": proj_h,
                    "accepted_calls_pct": 100 * (1 - total["rejected_call"] / max(total["calls"], 1)),
                    "pass": pct >= args.gate_changed and proj_h <= args.gate_hours}
            json.dump(gate, open(os.path.join(args.out, "pilot_gate.json"), "w"), indent=1)
            print("PILOT GATE", json.dumps(gate), flush=True)
            if not gate["pass"]:
                todo = todo[:done]
                break
    gen_s = time.time() - t1
    n_all_calls = sum(1 for e in exs if e.calls)
    stats = {"model": args.model, "examples": len(todo), "skipped_too_long": skipped, **total,
             "changed_pct": 100 * total["changed"] / max(total["args"], 1),
             "accepted_calls_pct": 100 * (1 - total["rejected_call"] / max(total["calls"], 1)),
             "load_s": load_s, "gen_s": gen_s, "tok_in": n_tok_in, "tok_out": n_tok_out,
             "projected_full_h": (gen_s / max(len(todo), 1)) * n_all_calls / 3600 + load_s / 3600}
    json.dump(stats, open(os.path.join(args.out, "distill_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    name = os.path.basename(args.inp).replace(".jsonl", "_distilled.jsonl")
    partial = args.limit or (args.pilot and len(todo) < sum(1 for e in exs if e.calls))
    save_jsonl(os.path.join(args.out, name), [exs[i] for i in todo] if partial else exs)
    open(os.path.join(args.out, "DONE"), "w").write("ok")


if __name__ == "__main__":
    main()
