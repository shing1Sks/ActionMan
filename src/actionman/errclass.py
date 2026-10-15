"""Error classes for wrong argument values (v3 analysis): garbled / truncated / list_format / span_length /
reasoning / missing / extra_arg / list_values / other. Reads an evaluate *_errors.jsonl and counts the class of every wrong argument
in calls whose tool name was right."""
from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter

from .scoring import _std, value_match


def _word_in(text: str, query: str) -> bool:
    return bool(text) and re.search(r"(?<!\w)" + re.escape(text.lower()) + r"(?!\w)", query.lower()) is not None


def classify_arg(pred, accept: list, query: str) -> str:
    acc = [a for a in accept if a != ""]
    if pred is None:
        return "missing"
    if not acc:
        return "extra_arg"                         # gold says this arg must be omitted
    if any(isinstance(a, list) for a in acc) and not isinstance(pred, list):
        return "list_format"
    if isinstance(pred, list) and any(isinstance(a, list) for a in acc):
        return "list_values"
    gs = [_std(a) for a in acc if not isinstance(a, (list, dict))]
    if isinstance(pred, (list, dict)) or not gs:
        return "other"
    ps, qs = _std(pred), _std(query)
    if _word_in(str(pred).strip(), query) and any(ps and s and (ps in s or s in ps) for s in gs):
        return "span_length"                       # copied a real phrase, but too short / too long
    if any(ps and s.startswith(ps) and ps != s for s in gs):
        return "truncated"
    gold_in_q = any(_word_in(str(a).strip(), query) for a in acc if not isinstance(a, (list, dict)))
    if gold_in_q and any(difflib.SequenceMatcher(None, ps, s).ratio() >= 0.6 for s in gs):
        return "garbled"                           # answer was in the query, output is a misspelling of it
    if not gold_in_q:
        return "reasoning"                         # answer is not in the query (conversion / knowledge / format)
    return "other"


def count(errors_path: str) -> Counter:
    c = Counter()
    for line in open(errors_path, encoding="utf-8"):
        r = json.loads(line)
        if not r["names_ok"]:
            c["wrong_tool"] += 1
            continue
        pool = {}
        for p in r["pred"]:
            pool.setdefault(p["name"], []).append(p["args"])
        for g in r["gold"]:
            (name, accept), = g.items()
            args = pool.get(name, [{}]).pop(0) if pool.get(name) else {}
            for k, acc in accept.items():
                pv = args.get(k)
                if pv is None and "" in acc:
                    continue
                if pv is not None and any(value_match(pv, a) for a in acc if a != ""):
                    continue
                c[classify_arg(pv, acc, r["query"])] += 1
    return c


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, dict(count(p).most_common()))
