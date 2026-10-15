"""ActionMan on BFCL v4 entries (single-turn): entry -> Example -> predict -> BFCL FC-style result.

Runs in ActionMan's own env (transformers 5.19); bench/bfcl_run.py (bfcl-eval venv) turns the output into result
files and scores them with the official checker.

python bench/bfcl_actionman.py --ckpt <ckpt> --entries data/bench/bfcl_v4_entries.jsonl --mode flat --out preds.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from actionman.schema import Example, from_bfcl, value_to_text  # noqa: E402

ROLE = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
SKIPPED = [{"__skipped__": "{}"}]   # decodes as a call, so a prompt that can't fit is wrong even on irrelevance


def category(entry_id: str) -> str:
    return entry_id.rsplit("_", 1)[0]


def flatten(messages: list[dict], mode: str = "flat") -> str:
    """flat: every message as 'Role: text', last user turn at the end (a lone user message stays bare).
    last: only the last user message (what schema.from_bfcl keeps)."""
    if mode == "last" or len(messages) == 1:
        users = [m for m in messages if m["role"] == "user"]
        return (users[-1] if users else messages[-1])["content"]
    return "\n".join(f"{ROLE.get(m['role'], m['role'].title())}: {m['content']}" for m in messages)


def entry_to_example(entry: dict, mode: str = "flat") -> Example:
    ex = from_bfcl(entry, None, category(entry["id"]))
    return dataclasses.replace(ex, query=flatten(entry["question"][0], mode), source=category(entry["id"]))


def fit(tok, ex: Example):
    """Truncate the query from the front until pass A fits MAX_LEN_A. -> (example, 'ok' | 'truncated' | 'too_long')."""
    from actionman.features import MAX_LEN_A, build_pass_a
    if build_pass_a(tok, ex) is not None:
        return ex, "ok"
    base = build_pass_a(tok, dataclasses.replace(ex, query=""))
    if base is None:
        return ex, "too_long"
    ids = tok(ex.query, add_special_tokens=False).input_ids
    keep = MAX_LEN_A - len(base["input_ids"])
    while keep > 0:
        short = dataclasses.replace(ex, query=tok.decode(ids[-keep:]).strip())
        if build_pass_a(tok, short) is not None:
            return short, "truncated"
        keep -= 8
    return ex, "too_long"


def to_result(calls, functions: list[dict], cat: str) -> list[dict]:
    """[{name: json args}] like an FC model. Values are typed by the declared schema: Java/JS args are source strings,
    Python 'string'/'any' args stay strings (text_to_value may have parsed '5' for an 'any' param)."""
    types = {f["name"]: {k: s.get("type") for k, s in (f.get("parameters") or {}).get("properties", {}).items()}
             for f in functions}
    out = []
    for c in calls:
        args = {}
        for k, v in c.args.items():
            if not isinstance(v, str) and (cat in ("simple_java", "simple_javascript")
                                           or types.get(c.name, {}).get(k) in ("string", "any")):
                v = value_to_text(v)
            args[k] = v
        out.append({c.name: json.dumps(args, ensure_ascii=False, default=_jsonable)})
    return out


def _jsonable(o):
    """text_to_value's literal_eval can yield sets/tuples ('{a, b}'); JSON has lists."""
    return sorted(o, key=str) if isinstance(o, (set, frozenset)) else list(o) if isinstance(o, tuple) else str(o)


# ---------- our 2,340-prompt eval in BFCL format (official-checker rescore) ----------

BFCL_TYPES = {"str": "string", "int": "integer", "float": "float", "bool": "boolean", "list": "array", "dict": "dict",
              "any": "any"}
PY_TYPES = {str: "string", int: "integer", float: "float", bool: "boolean", dict: "dict", list: "array"}


def tool_to_bfcl(tool, gold: list[dict]) -> dict:
    """Our Tool -> a BFCL function doc. Our schema keeps only 'list', so item types come from the gold values."""
    items = {}
    for g in gold:
        for k, vals in g.get(tool.name, {}).items():
            for v in vals:
                if isinstance(v, list) and v:
                    items.setdefault(k, PY_TYPES.get(type(v[0]), "any"))
    props = {}
    for p in tool.params:
        s = {"type": BFCL_TYPES[p.type], "description": p.description}
        if p.type == "list":
            s["items"] = {"type": items.get(p.name, "string")}
        if p.default is not None:
            s["default"] = p.default
        if p.options:
            s["enum"] = p.options
        props[p.name] = s
    return {"name": tool.name, "description": tool.description,
            "parameters": {"type": "dict", "properties": props, "required": [p.name for p in tool.params if not p.optional]}}


def _accept(v):
    """xLAM gold is exact-value; BFCL dict answers hold acceptable-value lists per key."""
    if isinstance(v, dict):
        return {k: [x] for k, x in v.items()}
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        return [_accept(x) for x in v]
    return v


def example_to_bfcl(ex: Example, v4_functions: dict | None = None):
    """-> (bfcl entry, possible-answer list, checker category). BFCL rows use the v4 function docs when given."""
    if ex.source.startswith("bfcl_"):
        cat = ex.source[5:]
        gold = ex.gold
        fns = (v4_functions or {}).get(ex.id.replace("simple_", "simple_python_", 1) if cat == "simple" else ex.id)
    else:
        cat = "irrelevance" if not ex.gold else "multiple" if len(ex.gold) == 1 else "parallel_multiple"
        gold = [{n: {k: [_accept(v) for v in vals] for k, vals in acc.items()} for n, acc in g.items()} for g in ex.gold]
        fns = None
    fns = fns or [tool_to_bfcl(t, gold) for t in ex.tools]
    entry = {"id": f"own_{ex.id}", "orig_id": ex.id, "source": ex.source,
             "question": [[{"role": "user", "content": ex.query}]], "function": fns}
    return entry, gold, cat


def predict_entry(model, tok, entry, mode, T, device, flags):
    from actionman.infer import predict
    ex, status = fit(tok, entry_to_example(entry, mode))
    if status == "too_long":
        return SKIPPED if "irrelevance" in ex.source else [], status, None
    pred = predict(model, tok, ex, T=T, device=device, **flags)
    return to_result([c.as_call() for c in pred.calls], entry["function"], ex.source), status, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--mode", default="flat", choices=["flat", "last"])
    ap.add_argument("--cats", default=None, help="comma list; default all in the file")
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="per category (smoke tests)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from actionman import beat
    from actionman.evaluate import is_cached, load_model, uses_markers, uses_v3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.ckpt, device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    flags = dict(cached=is_cached(args.ckpt), markers=uses_markers(args.ckpt), v3=uses_v3(args.ckpt))
    if args.compile and flags["cached"]:
        import actionman.cached as C
        C.encode_context, C.encode_slots = C.make_static(model.mlm.model)
    cats = set(args.cats.split(",")) if args.cats else None
    entries, seen = [], {}
    for line in open(args.entries, encoding="utf-8"):
        e = json.loads(line)
        c = category(e["id"])
        if (cats is None or c in cats) and (not args.limit or seen.get(c, 0) < args.limit):
            entries.append(e); seen[c] = seen.get(c, 0) + 1
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, e in enumerate(entries):
            if i % 50 == 0:
                beat(); print(f"{i}/{len(entries)} {time.time() - t0:.0f}s", flush=True)
            try:
                result, status, pred = predict_entry(model, tok, e, args.mode, args.T, device, flags)
            except Exception as ex:   # one bad prompt must not end the run; it is scored wrong
                print("error", e["id"], repr(ex)[:200], flush=True)
                result, status, pred = (SKIPPED if "irrelevance" in e["id"] else []), "error", None
            confs = [min([c.conf_tool] + list(c.conf_args.values())) for c in pred.calls] if pred else []
            f.write(json.dumps({"id": e["id"], "result": result, "status": status, "conf": min(confs) if confs else None,
                                "passes": pred.passes if pred else 0}, ensure_ascii=False) + "\n")
    print(f"done {len(entries)} in {time.time() - t0:.0f}s ->", args.out, flush=True)


if __name__ == "__main__":
    main()
