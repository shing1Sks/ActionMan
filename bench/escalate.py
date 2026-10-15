"""Escalation curve (v3 S5): ActionMan alone vs. handing low-confidence calls to the Qwen3-0.6B baseline.

Modes per confidence threshold theta (a call is shaky if min(tool conf, arg confs) < theta or an arg overflowed):
  none      ActionMan's answer as is
  full      shaky example -> Qwen from scratch with every tool (the usual LLM prompt)
  assisted  shaky example -> Qwen sees only ActionMan's chosen tools and its answer is pre-filled with every confident
            call plus the shaky call up to its first shaky value; Qwen writes only the rest

python bench/escalate.py --actionman <ckpt> --qwen <ckpt> --out escalate.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "baseline"))
from actionman import beat  # noqa: E402
from actionman.schema import Example  # noqa: E402


def call_conf(c) -> float:
    return min([c.conf_tool] + list(c.conf_args.values()))


def _shaky(c, theta) -> bool:
    return call_conf(c) < theta or bool(c.escalated)


def needs_escalation(pred, theta: float) -> bool:
    return any(_shaky(c, theta) for c in pred.calls)


def _tool_call(name, args) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def assisted_prefix(pred, ex: Example, theta: float):
    """(tools for Qwen's prompt, text that starts Qwen's answer) or None if nothing is shaky.
    An unsure tool choice gets the whole tool list and no prefix (Qwen redoes the example)."""
    if not needs_escalation(pred, theta):
        return None
    if any(c.conf_tool < theta for c in pred.calls):
        return list(ex.tools), ""
    good = [c for c in pred.calls if not _shaky(c, theta)]
    bad = [c for c in pred.calls if _shaky(c, theta)]
    by_name = {t.name: t for t in ex.tools}
    tools = []
    for c in good + bad:
        if by_name[c.name] not in tools:
            tools.append(by_name[c.name])
    s = bad[0]
    keep = {k: v for k, v in s.args.items() if s.conf_args.get(k, 1.0) >= theta}
    first_bad = next((k for k in s.args if k not in keep), s.escalated[0] if s.escalated else None)
    head = json.dumps({"name": s.name, "arguments": keep}, ensure_ascii=False)[:-2]
    if first_bad is not None:
        head += (", " if keep else "") + json.dumps(first_bad) + ": "
    return tools, "\n".join([_tool_call(c.name, c.args) for c in good] + ["<tool_call>\n" + head])


def finish(prefix: str, continuation: str):
    from qwen import parse_calls
    return parse_calls(prefix + continuation)


# ---------- GPU part ----------

def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def generate(model, tok, prompts: list[str], device, batch=16, max_new=256) -> list[str]:
    tok.padding_side = "left"
    out = []
    for s in range(0, len(prompts), batch):
        enc = tok(prompts[s: s + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=False)
        beat()
    return out


def score(rows, key):
    from actionman.scoring import match_calls
    st = collections.defaultdict(collections.Counter)
    for r in rows:
        ok = match_calls(r[key], r["ex"].gold)
        for k in (r["ex"].source, "all"):
            st[k]["n"] += 1
            st[k]["ok"] += ok
    return {k: round(v["ok"] / v["n"], 4) for k, v in st.items()}


def stats(ms):
    ms = sorted(ms)
    return {"p50": round(statistics.median(ms), 1), "mean": round(sum(ms) / len(ms), 1),
            "p95": round(ms[int(0.95 * (len(ms) - 1))], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actionman", required=True)
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--data", nargs="+", default=["data/prepared/test.jsonl", "data/prepared/bfcl.jsonl"])
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--thetas", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.85])
    ap.add_argument("--n-latency", type=int, default=300)
    ap.add_argument("--latency-data", default="data/prepared/bfcl.jsonl")
    ap.add_argument("--out", default="escalate.json")
    args = ap.parse_args()
    from actionman.data import load_jsonl
    from actionman.evaluate import is_cached, load_model, uses_markers, uses_v3
    from actionman.infer import predict
    from qwen import load as qload, parse_calls, prompt_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    am, atok = load_model(args.actionman, device)
    if device == "cuda":
        am = am.to(torch.bfloat16)
    kw = dict(T=args.T, device=device, cached=is_cached(args.actionman), markers=uses_markers(args.actionman),
              v3=uses_v3(args.actionman))
    qm, qtok = qload(args.qwen, device=device)
    qm.eval()
    exs = [e for p in args.data for e in load_jsonl(p)]

    # accuracy
    rows = []
    for n, ex in enumerate(exs):
        if n % 50 == 0:
            beat()
        p = predict(am, atok, ex, **kw)
        rows.append({"ex": ex, "pred": p, "none": [c.as_call() for c in p.calls]})
    esc_any = [r for r in rows if needs_escalation(r["pred"], max(args.thetas))]
    full_txt = generate(qm, qtok, [prompt_text(qtok, r["ex"]) for r in esc_any], device)
    for r, t in zip(esc_any, full_txt):
        r["qwen_full"] = parse_calls(t)
    result = {"n": len(rows), "actionman_only": score(rows, "none"), "thetas": {}}
    for th in args.thetas:
        jobs = []
        for r in rows:
            pre = assisted_prefix(r["pred"], r["ex"], th)
            r[f"full@{th}"] = r["qwen_full"] if pre is not None else r["none"]
            r[f"assisted@{th}"] = r["none"]
            if pre is not None:
                tools, prefix = pre
                ex_t = Example(r["ex"].query, tools, [], r["ex"].id, r["ex"].source)
                jobs.append((r, prefix, prompt_text(qtok, ex_t) + prefix))
        conts = generate(qm, qtok, [j[2] for j in jobs], device)
        for (r, prefix, _), c in zip(jobs, conts):
            r[f"assisted@{th}"] = finish(prefix, c)
        result["thetas"][str(th)] = {"escalation_rate": round(len(jobs) / len(rows), 4),
                                    "full": score(rows, f"full@{th}"), "assisted": score(rows, f"assisted@{th}")}
        print(th, json.dumps(result["thetas"][str(th)]), flush=True)

    # batch-1 latency on the same GPU
    lat_ex = load_jsonl(args.latency_data)[: args.n_latency]
    for ex in lat_ex[:10]:                                       # warm-up
        predict(am, atok, ex, **kw)
        generate(qm, qtok, [prompt_text(qtok, ex)], device, batch=1)
    t_am, t_q, t_ass, t_full = [], [], collections.defaultdict(list), collections.defaultdict(list)
    for n, ex in enumerate(lat_ex):
        if n % 20 == 0:
            beat()
        _sync(device); t0 = time.perf_counter()
        p = predict(am, atok, ex, **kw)
        _sync(device); a_ms = 1000 * (time.perf_counter() - t0)
        t_am.append(a_ms)
        t0 = time.perf_counter()
        generate(qm, qtok, [prompt_text(qtok, ex)], device, batch=1)
        _sync(device); q_ms = 1000 * (time.perf_counter() - t0)
        t_q.append(q_ms)
        for th in (0.5, 0.7):
            pre = assisted_prefix(p, ex, th)
            if pre is None:
                t_ass[th].append(a_ms); t_full[th].append(a_ms)
                continue
            tools, prefix = pre
            t0 = time.perf_counter()
            generate(qm, qtok, [prompt_text(qtok, Example(ex.query, tools, [], ex.id, ex.source)) + prefix], device, batch=1)
            _sync(device)
            t_ass[th].append(a_ms + 1000 * (time.perf_counter() - t0))
            t_full[th].append(a_ms + q_ms)
    result["latency_ms"] = {"actionman_only": stats(t_am), "qwen_only": stats(t_q),
                            **{f"assisted@{th}": stats(v) for th, v in t_ass.items()},
                            **{f"full@{th}": stats(v) for th, v in t_full.items()}}
    print(json.dumps(result["latency_ms"]), flush=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
