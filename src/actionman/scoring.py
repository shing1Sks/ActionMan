"""BFCL-style AST matching (re-implemented, simplified) for predicted calls vs acceptable-value gold.

gold: [{func_name: {arg: [acceptable values]}}]; "" in the acceptable list means the arg may be omitted.
"""
from __future__ import annotations

import itertools
import re
from typing import Any

from .schema import Call


def _std(s: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", str(s)).lower().replace("'", '"')


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def value_match(pred: Any, gold: Any) -> bool:
    if isinstance(gold, bool) or isinstance(pred, bool):
        if isinstance(pred, str):
            return pred.strip().lower() == str(gold).lower()
        return pred == gold
    gn, pn = _num(gold), _num(pred)
    if gn is not None and not isinstance(gold, str):
        return pn is not None and abs(pn - gn) < 1e-6
    if isinstance(gold, str):
        return _std(pred) == _std(gold) if not isinstance(pred, (list, dict)) else False
    if isinstance(gold, (list, tuple)):
        return isinstance(pred, (list, tuple)) and len(pred) == len(gold) and all(
            value_match(p, g) for p, g in zip(pred, gold))
    if isinstance(gold, dict):
        if not isinstance(pred, dict):
            return False
        for k, gv in gold.items():
            accept = gv if isinstance(gv, list) else [gv]
            if k not in pred:
                if "" not in accept:
                    return False
                continue
            if not any(value_match(pred[k], a) for a in accept if a != ""):
                return False
        return set(pred) <= set(gold)
    return pred == gold


def call_match(pred: Call, gold_call: dict) -> bool:
    (name, accept), = gold_call.items()
    if pred.name != name:
        return False
    for k, v in pred.args.items():
        if k not in accept:
            return False
        if not any(value_match(v, a) for a in accept[k] if not (isinstance(a, str) and a == "")):
            if not ("" in accept[k] and v in ("", None)):
                return False
    for k, vals in accept.items():
        if k not in pred.args and "" not in vals:
            return False
    return True


def match_calls(pred: list[Call], gold: list[dict]) -> bool:
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    for perm in itertools.permutations(range(len(pred))):
        if all(call_match(pred[perm[g]], gold[g]) for g in range(len(gold))):
            return True
    return False


def names_match(pred: list[Call], gold: list[dict]) -> bool:
    return sorted(c.name for c in pred) == sorted(next(iter(g)) for g in gold)
