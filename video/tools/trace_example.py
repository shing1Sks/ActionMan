"""Trace one real ActionMan prediction pass by pass, for the mechanism scene of the film (CPU, no GPU needed).

Runs the release-candidate checkpoint on a BFCL item with the exact v3 inference logic (`actionman.infer_v3.predict_v3`,
copied here step by step with a recorder added) and writes video/trace.json:
tool probabilities per [CALL] slot, length buckets, the slot block, copy gates and spans, every slot token after each
refinement pass, option probabilities, and the final call. It also tokenises the same gold call the way the Qwen3-0.6B
baseline would write it (Hermes <tool_call> JSON) so the cold open types the LLM's real tokens.

  PYTHONPATH=src .venv/Scripts/python.exe video/tools/trace_example.py --ckpt runs/hf/runs/s16-v3-5epb/m --id multiple_137
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from actionman.data import load_jsonl  # noqa: E402
from actionman.evaluate import load_model  # noqa: E402
from actionman.features import BUCKETS, MAX_CALLS, OVERFLOW  # noqa: E402
from actionman.infer import _banned_ids  # noqa: E402
from actionman.schema import Example, param_options, signature, value_to_text  # noqa: E402
from actionman.span import best_span  # noqa: E402
from actionman.v3feat import build_pass_a_v3, build_pass_b_v3  # noqa: E402


@torch.no_grad()
def trace(model, tok, ex: Example, T=3, tau=0.85):
    rec = {"id": ex.id, "query": ex.query, "T": T, "tau": tau,
           "tools": [{"name": t.name, "sig": signature(t), "params": [[p.name, p.type, p.optional] for p in t.params]}
                     for t in ex.tools]}
    ex0 = Example(ex.query, ex.tools, [], ex.id, ex.source)
    a = build_pass_a_v3(tok, ex0)
    ids = torch.tensor([a["input_ids"]])
    rec["context_tokens"] = len(a["input_ids"])
    from actionman.cached import encode_context, encode_slots
    t0 = time.perf_counter()
    h, kv = encode_context(model.mlm.model, ids)
    J = len(a["t_pos"])
    probs = model.decision_logits(h, torch.tensor([a["call_pos"]]), torch.tensor([a["t_pos"]]),
                                  torch.ones(1, J, dtype=torch.bool)).float().softmax(-1)[0]
    rec["call_slots"] = [[round(float(p), 4) for p in probs[i]] for i in range(MAX_CALLS)]   # last column = NONE
    chosen = []
    for i in range(MAX_CALLS):
        j = int(probs[i].argmax())
        if j == J:
            break
        chosen.append((i, j, float(probs[i, j])))
    rows, keys = [], []
    for i, j, _ in chosen:
        for k in range(len(ex.tools[j].params)):
            rows.append((0, a["call_pos"][i], a["arg_pos"][(j, k)]))
            keys.append((i, j, k))
    idx = torch.tensor(rows).reshape(-1, 3)
    len_logits = model.length_logits(h, idx).float()
    cnt_logits = model.count_logits(h, idx).float()
    plan, rec["calls"] = [], []
    for i, j, conf in chosen:
        modes, params = {}, []
        for r, (ri, _, k) in enumerate(keys):
            if ri != i:
                continue
            p = ex.tools[j].params[k]
            lg = len_logits[r].clone()
            if not p.optional:
                lg[0] = float("-inf")
            b = int(lg.argmax())
            info = {"name": p.name, "type": p.type, "bucket": b, "bucket_slots": BUCKETS[b] if b < len(BUCKETS) else None,
                    "bucket_p": round(float(lg.softmax(-1)[b]), 4)}
            if b != 0:
                if param_options(p) is not None:
                    modes[k] = ("opt", 0); info["mode"] = "option"
                elif p.type == "list" and int(cnt_logits[r].argmax()) >= 1:
                    modes[k] = ("list", int(cnt_logits[r].argmax())); info["mode"] = "list"; info["items"] = modes[k][1]
                else:
                    modes[k] = ("text", b); info["mode"] = "text"
                    if b == OVERFLOW:
                        info["mode"] = "overflow"
            else:
                info["mode"] = "absent"
            params.append(info)
        plan.append((j, modes))
        rec["calls"].append({"slot": i, "tool": ex.tools[j].name, "p": round(conf, 4), "params": params})

    bb = build_pass_b_v3(tok, ex0, a, plan, with_targets=False, keep_context=True, markers=True)
    ids_b = torch.tensor([bb["input_ids"]])
    c0 = bb["ctx_len"]
    rec["block_tokens"] = [tok.convert_ids_to_tokens(int(x)) for x in bb["input_ids"][c0:]]
    rec["slot_pos_rel"] = [[p - c0 for p in pos] for pos in bb["slot_pos"]]
    rec["slot_index"] = [list(k) if k[2] is not None else [k[0], k[1], None] for k in bb["slot_index"]]
    rec["opt_slots_rel"] = [{"call": s[0], "param": s[1], "anchor": s[2] - c0, "opts": [o - c0 for o in s[3]]} for s in bb["opt_slots"]]

    flat = [p for pos in bb["slot_pos"] for p in pos]
    fpos = torch.tensor(flat, dtype=torch.long)
    conf = torch.zeros(len(flat))
    masked = torch.ones(len(flat), dtype=torch.bool)
    fixed = torch.zeros(len(flat), dtype=torch.bool)
    banned = _banned_ids(tok, model.mlm.decoder.weight.shape[0])
    starts = [0]
    for pos in bb["slot_pos"]:
        starts.append(starts[-1] + len(pos))
    rec["copies"], rec["passes"] = [], []
    hb = None
    for t in range(T):
        hs = encode_slots(model.mlm.model, ids_b[:, c0:], c0, kv)
        hb = torch.cat([h, hs], dim=1)
        if t == 0 and bb["anchors"]:
            ab = torch.zeros(len(bb["anchors"]), dtype=torch.long)
            ap = torch.tensor(bb["anchors"])
            qpos = torch.tensor([a["q_pos"]]).expand(len(bb["anchors"]), -1)
            g, st, en = model.copy_logits(hb, ab, ap, qpos)
            g = g.float().sigmoid()
            for v in range(len(bb["anchors"])):
                c = {"value": v, "gate": round(float(g[v]), 4), "copy": False}
                if g[v] > 0.5 and a["q_pos"]:
                    s, e, pspan = best_span(st[v], en[v])
                    cs, ce = a["q_offs"][s][0], a["q_offs"][e][1]
                    text = ex.query[cs:ce].strip()
                    if text:
                        c.update(copy=True, text=text, char_start=cs, char_end=ce, span_p=round(pspan, 4))
                        n = len(bb["slot_pos"][v])
                        span_ids = [a["input_ids"][q] for q in a["q_pos"][s: e + 1]][:n]
                        span_ids += [tok.pad_token_id] * (n - len(span_ids))
                        sl = slice(starts[v], starts[v + 1])
                        ids_b[0, fpos[sl]] = torch.tensor(span_ids)
                        fixed[sl] = True
                rec["copies"].append(c)
        if not flat:
            break
        logits = model.mlm.decoder(model.mlm.head(hb[0, fpos])).float()
        logits[:, banned] = float("-inf")
        p, tokens = logits.softmax(-1).max(-1)
        upd = masked & ~fixed
        conf = torch.where(upd, p, conf)
        ids_b[0, fpos[upd]] = tokens[upd]
        conf = torch.where(fixed, torch.ones_like(conf), conf)
        step = {"pass": t + 1, "updated": upd.tolist(),
                "tokens": [tok.convert_ids_to_tokens(int(x)) for x in ids_b[0, fpos]],
                "conf": [round(float(x), 4) for x in conf], "fixed": fixed.tolist()}
        rec["passes"].append(step)
        if t == T - 1:
            break
        masked = (conf < tau) & ~fixed
        step["remasked"] = masked.tolist()
        if not masked.any():
            break
        ids_b[0, fpos[masked]] = tok.mask_token_id
    rec["options"] = []
    if bb["opt_slots"]:
        ob = torch.zeros(len(bb["opt_slots"]), dtype=torch.long)
        oa = torch.tensor([s[2] for s in bb["opt_slots"]])
        M = max(len(s[3]) for s in bb["opt_slots"])
        opos = torch.full((len(bb["opt_slots"]), M), -1, dtype=torch.long)
        for n, s in enumerate(bb["opt_slots"]):
            opos[n, : len(s[3])] = torch.tensor(s[3])
        po = model.option_logits(hb, ob, oa, opos).float().softmax(-1)
        for n, (i, k, _, opts, _) in enumerate(bb["opt_slots"]):
            j = chosen[[c[0] for c in chosen].index(i)][1]
            o = param_options(ex.tools[j].params[k])
            rec["options"].append({"call": i, "param": ex.tools[j].params[k].name,
                                   "options": [value_to_text(x) for x in o], "p": [round(float(x), 4) for x in po[n, : len(o)]]})
    rec["cpu_ms"] = round((time.perf_counter() - t0) * 1000)
    rec["forward_passes"] = 1 + len(rec["passes"])
    # decoded values, same rules as predict_v3
    vals = []
    for v, pos in enumerate(bb["slot_pos"]):
        cp = next((c for c in rec["copies"] if c["value"] == v and c["copy"]), None)
        if cp:
            vals.append(cp["text"]); continue
        toks = ids_b[0, fpos[starts[v]: starts[v + 1]]].tolist()
        if tok.pad_token_id in toks:
            toks = toks[: toks.index(tok.pad_token_id)]
        vals.append(tok.decode(toks, skip_special_tokens=True).strip())
    rec["values"] = vals
    rec["gold"] = [{"name": c.name, "args": c.args} for c in ex.calls]
    return rec


def qwen_tokens(ex: Example):
    """The same gold call in the Qwen baseline's output format (baseline/qwen.py), split into its real tokens."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    text = "\n".join("<tool_call>\n" + json.dumps({"name": c.name, "arguments": c.args}, ensure_ascii=False) + "\n</tool_call>"
                     for c in ex.calls) + "<|im_end|>"
    ids = tk(text, add_special_tokens=False).input_ids
    return [tk.decode([i]) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/hf/runs/s16-v3-5epb/m"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/prepared/bfcl.jsonl"))
    ap.add_argument("--id", nargs="+", default=["multiple_137"])
    ap.add_argument("--out", default=os.path.join(ROOT, "video/trace.json"))
    ap.add_argument("--scan", type=int, default=0, help="instead: trace the first N items of --data and list candidates")
    ap.add_argument("--prefix", default="multiple_")
    args = ap.parse_args()
    torch.manual_seed(0)
    model, tok = load_model(args.ckpt, "cpu")
    if args.scan:
        n = 0
        for e in load_jsonl(args.data):
            if not e.id.startswith(args.prefix) or len(e.calls) != 1 or len(e.query) > 120:
                continue
            if all(value_to_text(v).lower() in e.query.lower() for v in e.calls[0].args.values()):
                continue                              # want at least one value that must be written, not copied
            r = trace(model, tok, e)
            modes = [p["mode"] for c in r["calls"] for p in c["params"]]
            gold = [value_to_text(v) for v in e.calls[0].args.values()]
            print(e.id, "slot passes", len(r["passes"]), modes, r["values"], "gold", gold, flush=True)
            n += 1
            if n >= args.scan:
                return
    exs = {e.id: e for e in load_jsonl(args.data) if e.id in args.id}
    out = {"_source": f"video/tools/trace_example.py on {os.path.relpath(args.ckpt, ROOT)} (CPU, fp32)", "examples": {}}
    for i in args.id:
        rec = trace(model, tok, exs[i])
        rec["qwen_tokens"] = qwen_tokens(exs[i])
        out["examples"][i] = rec
        print(i, rec["values"], "passes", rec["forward_passes"], "cpu", rec["cpu_ms"], "ms", "qwen tokens", len(rec["qwen_tokens"]))
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
