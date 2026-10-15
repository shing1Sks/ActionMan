"""Evaluate an ActionMan checkpoint on prepared jsonl files; writes metrics JSON."""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

import torch

from .data import load_jsonl
from .features import load_tokenizer
from .infer import predict
from . import beat
from .scoring import match_calls, names_match


def ece(pairs, bins=10):
    if not pairs:
        return 0.0
    tot, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, y) for c, y in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            err += len(sel) / tot * abs(sum(c for c, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
    return err


def evaluate(model, tok, examples, T=2, tau=0.85, device="cpu", limit=None, cached=False, errors=None, markers=False,
             v3=False, snap=False, dump=None):
    stats = collections.defaultdict(lambda: collections.Counter())
    conf_pairs, passes, t0 = [], [], time.time()
    for ex in examples[:limit] if limit else examples:
        if len(passes) % 50 == 0:
            beat()
        pred = predict(model, tok, ex, T=T, tau=tau, device=device, cached=cached, markers=markers, v3=v3, snap=snap)
        calls = [c.as_call() for c in pred.calls]
        full = match_calls(calls, ex.gold)
        name = names_match(calls, ex.gold)
        cat = ex.source
        for key in (cat, "all"):
            s = stats[key]
            s["n"] += 1; s["full"] += full; s["names"] += name; s["skipped"] += pred.skipped
            s["escalated"] += any(c.escalated for c in pred.calls)
        passes.append(pred.passes)
        confs = [c.conf_tool for c in pred.calls] + [v for c in pred.calls for v in c.conf_args.values()]
        conf_pairs.append((min(confs) if confs else 1.0, float(full)))
        if (errors is not None and not full) or dump is not None:
            rec = {"id": ex.id, "source": ex.source, "query": ex.query, "names_ok": name,
                   "pred": [{"name": c.name, "args": c.args, "conf_tool": round(c.conf_tool, 3),
                             "conf_args": {k: round(v, 3) for k, v in c.conf_args.items()},
                             "escalated": c.escalated} for c in pred.calls],
                   "gold": ex.gold}
            if errors is not None and not full:
                errors.append(rec)
            if dump is not None:   # every prediction (official-checker rescore, calibration, coverage)
                dump.append({**rec, "full": bool(full), "skipped": pred.skipped})
    out = {k: {"n": s["n"], "full_acc": s["full"] / s["n"], "tool_acc": s["names"] / s["n"],
               "skipped": s["skipped"], "escalated": s["escalated"]} for k, s in stats.items()}
    out["_meta"] = {"T": T, "tau": tau, "cached": cached, "avg_passes": sum(passes) / max(len(passes), 1),
                    "ece": ece(conf_pairs), "seconds": time.time() - t0}
    return out


def is_cached(ckpt: str) -> bool:
    return bool(json.load(open(os.path.join(ckpt, "run.json"))).get("cached", False))


def uses_markers(ckpt: str) -> bool:
    return bool(json.load(open(os.path.join(ckpt, "run.json"))).get("markers", False))


def uses_v3(ckpt: str) -> bool:
    return bool(json.load(open(os.path.join(ckpt, "run.json"))).get("v3", False))


def load_model(ckpt: str, device="cpu"):
    from transformers import ModernBertConfig
    from .model import ActionMan
    tok = load_tokenizer(json.load(open(os.path.join(ckpt, "run.json")))["tokenizer"])
    cfg = ModernBertConfig.from_json_file(os.path.join(ckpt, "encoder_config.json"))
    model = ActionMan.from_encoder(cfg, tok, v3=uses_v3(ckpt))
    sd = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    return model.to(device).eval(), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", default=["data/prepared/test.jsonl", "data/prepared/bfcl.jsonl"])
    ap.add_argument("--T", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compile", action="store_true", help="cached models: padded static encoders under torch.compile")
    ap.add_argument("--snap", action="store_true", help="v3: widen copied spans to whole words/numbers")
    ap.add_argument("--dump", action="store_true", help="also write every prediction (largest T) to <out>_preds.jsonl")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.ckpt, device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    if args.compile and is_cached(args.ckpt):
        import actionman.cached as C
        C.encode_context, C.encode_slots = C.make_static(model.mlm.model)
    exs = [e for p in args.data for e in load_jsonl(p)]
    results, errs, dump = {}, {}, ([] if args.dump else None)
    for T in args.T:
        errs[T] = []
        results[f"T{T}"] = evaluate(model, tok, exs, T, args.tau, device, args.limit, is_cached(args.ckpt), errs[T],
                                     markers=uses_markers(args.ckpt), v3=uses_v3(args.ckpt), snap=args.snap,
                                     dump=dump if T == max(args.T) else None)
    out = args.out or os.path.join(args.ckpt, "eval.json")
    json.dump(results, open(out, "w"), indent=1)
    with open(out.replace(".json", "_errors.jsonl"), "w", encoding="utf-8") as f:
        for e in errs[max(args.T)]:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + chr(10))
    if dump is not None:
        with open(out.replace(".json", "_preds.jsonl"), "w", encoding="utf-8") as f:
            for e in dump:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + chr(10))
    for k, r in results.items():
        print(k, json.dumps({c: (round(v["full_acc"], 3), round(v["tool_acc"], 3)) for c, v in r.items() if c != "_meta"}),
              r["_meta"])


if __name__ == "__main__":
    main()
