"""Data audit (v3 S0): how much argument mass can be copied from the query, picked from options, or is a list."""
from __future__ import annotations

import os
from collections import Counter, defaultdict

from .schema import param_options, value_to_text


def query_offsets(tok, query: str) -> list[tuple[int, int]]:
    """Char span per query token, with a leading space (BPE 'Ġ') trimmed off."""
    offs = tok(query, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    out = []
    for a, b in offs:
        while a < b and query[a].isspace():
            a += 1
        out.append((a, b))
    return out


def copy_span(tok, query: str, text: str, offs=None):
    """First case-insensitive, token-aligned occurrence of `text` in `query` -> (start_tok, end_tok) or None."""
    text = str(text).strip()
    if not text:
        return None
    offs = offs if offs is not None else query_offsets(tok, query)
    starts = {a: i for i, (a, b) in reversed(list(enumerate(offs))) if b > a}
    ends = {b: i for i, (a, b) in enumerate(offs) if b > a}
    low, t = query.lower(), text.lower()
    at = low.find(t)
    while at != -1:
        s, e = starts.get(at), ends.get(at + len(t))
        if s is not None and e is not None and e >= s:
            return s, e
        at = low.find(t, at + 1)
    return None


def _kind(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "num"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return "other"


def audit(tok, examples) -> dict:
    """Per-source counters over every gold argument value."""
    stats: dict[str, Counter] = defaultdict(Counter)
    for ex in examples:
        c = stats[ex.source.split("_hardneg")[0].split("_neg")[0] if ex.source.startswith("xlam") else ex.source]
        if not ex.calls:
            continue
        offs = query_offsets(tok, ex.query)
        by_name = {t.name: t for t in ex.tools}
        for call in ex.calls:
            tool = by_name.get(call.name)
            if tool is None:
                continue
            pmap = {p.name: p for p in tool.params}
            for name, v in call.args.items():
                p = pmap.get(name)
                k = _kind(v)
                c["args"] += 1
                c[f"type_{k}"] += 1
                n_tok = len(tok(value_to_text(v), add_special_tokens=False).input_ids)
                if n_tok > 32:
                    c["over_32_tokens"] += 1
                opts = param_options(p) if p else None
                if opts is not None:
                    c["has_options"] += 1
                    if any(value_to_text(o).lower() == value_to_text(v).lower() for o in opts):
                        c["gold_in_options"] += 1
                if k in ("str", "num"):
                    c[f"{k}_total"] += 1
                    if copy_span(tok, ex.query, value_to_text(v), offs) is not None:
                        c[f"{k}_copyable"] += 1
                if k == "list":
                    c["list_items"] += len(v)
                    if len(v) > 7:
                        c["list_over_7"] += 1
                    for it in v:
                        c["list_item_total"] += 1
                        if len(tok(value_to_text(it), add_special_tokens=False).input_ids) > 8:
                            c["list_item_over_8_tokens"] += 1
                        if isinstance(it, (str, int, float)) and not isinstance(it, bool) and \
                                copy_span(tok, ex.query, value_to_text(it), offs) is not None:
                            c["list_item_copyable"] += 1
    return stats


def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def report(stats: dict) -> str:
    rows = ["| source | args | str copyable | num copyable | bool/enum (options) | gold in options | list args | "
            "list items copyable | item >8 tok | list >7 | value >32 tok |", "|" + "---|" * 11]
    for src in sorted(stats):
        c = stats[src]
        if not c["args"]:
            continue
        rows.append(f"| {src} | {c['args']} | {_pct(c['str_copyable'], c['str_total'])} of {c['str_total']} | "
                    f"{_pct(c['num_copyable'], c['num_total'])} of {c['num_total']} | "
                    f"{_pct(c['has_options'], c['args'])} | {_pct(c['gold_in_options'], c['has_options'])} | "
                    f"{_pct(c['type_list'], c['args'])} | {_pct(c['list_item_copyable'], c['list_item_total'])} | "
                    f"{_pct(c['list_item_over_8_tokens'], c['list_item_total'])} | {c['list_over_7']} | "
                    f"{_pct(c['over_32_tokens'], c['args'])} |")
    return "\n".join(rows)


if __name__ == "__main__":
    from .data import load_jsonl
    from .features import load_tokenizer
    tok = load_tokenizer()
    parts = []
    for f in ("train_v2.jsonl", "test.jsonl", "bfcl.jsonl"):
        st = audit(tok, load_jsonl(os.path.join("data/prepared", f)))
        parts.append(f"## {f}\n\n{report(st)}\n")
    out = "# v3 data audit (S0)\n\nShares are over gold argument values. 'copyable' = the value text occurs in the " \
          "query, case-insensitively, on token boundaries.\n\n" + "\n".join(parts)
    open("docs/results/audit.md", "w", encoding="utf-8").write(out)
    print(out)
