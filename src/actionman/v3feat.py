"""v3 sequences: copy targets into the query, Jev-style option blocks, per-item list slots.

Pass B v3 slot block:
  #1 tool( city= [M][M][M][M] , unit= [M] ( [OPT] metric [OPT] imperial ) , tags= [ [M]x8 | [M]x8 ] )
Every scalar value / list item has an anchor (its first slot) that carries the copy gate and the span pointer;
every option param has one [MASK] anchor that points at its [OPT] markers. The v2 builders are untouched.
"""
from __future__ import annotations

import torch

from .audit import copy_span, query_offsets
from .features import BUCKETS, MAX_CALLS, MAX_LEN_B, OVERFLOW, _enc, build_pass_a, value_tokens
from .schema import Example, param_options, value_to_text
from .scoring import value_match

ITEM_SLOTS = 8
MAX_ITEMS = 7
N_COUNT = MAX_ITEMS + 1          # count classes: 0 = whole list as text (v2 path), 1..7 = item slots


def _scalar(v) -> bool:
    return isinstance(v, (str, int, float)) and not isinstance(v, bool)


def list_mode(tok, query: str, v, offs) -> bool:
    """A list gets item slots if it has 1..7 scalar items, each copyable or at most ITEM_SLOTS tokens."""
    if not isinstance(v, list) or not 1 <= len(v) <= MAX_ITEMS:
        return False
    for it in v:
        if not _scalar(it):
            return False
        if len(value_tokens(tok, it)) > ITEM_SLOTS and copy_span(tok, query, value_to_text(it), offs) is None:
            return False
    return True


def option_index(p, v) -> int | None:
    opts = param_options(p)
    if opts is None:
        return None
    for m, o in enumerate(opts):
        if value_match(v, o) or value_to_text(v).strip().lower() == value_to_text(o).strip().lower():
            return m
    return None


def build_pass_a_v3(tok, ex: Example) -> dict | None:
    """Pass A plus query token positions/char offsets and count labels for declared-list params."""
    a = build_pass_a(tok, ex)
    if a is None:
        return None
    offs = query_offsets(tok, ex.query)
    q_end = a["call_pos"][0]                       # query tokens sit right before the [CALL] markers
    a["q_pos"] = list(range(q_end - len(offs), q_end))
    a["q_offs"] = offs
    count_items = []
    for (i, j, k, b) in a["len_items"]:
        p = ex.tools[j].params[k]
        if p.type == "list" and b != 0:
            v = ex.calls[i].args[p.name]
            count_items.append((i, j, k, len(v) if list_mode(tok, ex.query, v, offs) else 0))
    a["count_items"] = count_items
    return a


def gold_plan_v3(tok, ex: Example, a: dict) -> list[tuple[int, dict[int, tuple[str, int]]]]:
    """Per gold call: (tool_j, {param_k: (mode, n)}) with mode in opt / list (n items) / text (n = bucket)."""
    plan = []
    for i in range(min(len(ex.calls), MAX_CALLS)):
        j, modes = a["dec_labels"][i], {}
        for (ci, _, k, b) in a["len_items"]:
            if ci != i or b == 0:
                continue
            p = ex.tools[j].params[k]
            v = ex.calls[i].args[p.name]
            if option_index(p, v) is not None:
                modes[k] = ("opt", 0)
            elif p.type == "list" and list_mode(tok, ex.query, v, a["q_offs"]):
                modes[k] = ("list", len(v))
            else:
                modes[k] = ("text", b)
        plan.append((j, modes))
    return plan


def build_pass_b_v3(tok, ex: Example, a: dict, plan, with_targets: bool, max_len: int = MAX_LEN_B,
                    keep_context: bool = False, markers: bool = True) -> dict | None:
    ids = (a["input_ids"] if keep_context else a["input_ids"][:-1]) + [tok.am["[ARGS]"]]
    ctx_len = len(a["input_ids"]) if keep_context else len(ids) - 1
    out = {"slot_pos": [], "slot_index": [], "anchors": [], "opt_slots": [], "ctx_len": ctx_len, "q_pos": a["q_pos"]}
    if with_targets:
        out["slot_targets"], out["copy_targets"] = [], []

    def add_value(n, value, key):
        pos = list(range(len(ids), len(ids) + n))
        ids.extend([tok.mask_token_id] * n)
        out["slot_pos"].append(pos)
        out["slot_index"].append(key)
        out["anchors"].append(pos[0])
        if with_targets:
            v = value_tokens(tok, value)[:n]
            out["slot_targets"].append(v + [tok.pad_token_id] * (n - len(v)))
            cs = copy_span(tok, ex.query, value_to_text(value), a["q_offs"]) if _scalar(value) else None
            out["copy_targets"].append((a["q_pos"][cs[0]], a["q_pos"][cs[1]]) if cs else None)

    for i, (j, modes) in enumerate(plan):
        tool = ex.tools[j]
        ids.extend(_enc(tok, (f" #{i + 1} " if markers else "") + tool.name + "("))
        first = True
        for k, p in enumerate(tool.params):
            if k not in modes:
                continue
            mode, n = modes[k]
            if mode == "text" and n in (0, OVERFLOW):
                continue
            ids.extend(_enc(tok, ("" if first else ", ") + p.name + "="))
            first = False
            v = ex.calls[i].args[p.name] if with_targets else None
            if mode == "opt":
                anchor = len(ids)
                ids.append(tok.mask_token_id)
                ids.extend(_enc(tok, " ("))
                opos = []
                for o in param_options(p):
                    opos.append(len(ids))
                    ids.append(tok.am["[OPT]"])
                    ids.extend(_enc(tok, " " + value_to_text(o)))
                ids.extend(_enc(tok, " )"))
                tgt = option_index(p, v) if with_targets else None
                out["opt_slots"].append((i, k, anchor, opos, -100 if tgt is None else tgt))
            elif mode == "list":
                ids.extend(_enc(tok, " ["))
                for it in range(n):
                    if it:
                        ids.extend(_enc(tok, " |"))
                    add_value(ITEM_SLOTS, v[it] if with_targets else None, (i, k, it))
                ids.extend(_enc(tok, " ]"))
            else:
                add_value(BUCKETS[n], v, (i, k, None))
        ids.extend(_enc(tok, ")"))
    ids.append(tok.sep_token_id)
    if len(ids) > max_len:
        return None
    out["input_ids"] = ids
    return out


# ---------- batching ----------

def v3_extras(A: list[dict], B_rows: list[tuple[int, dict]]) -> dict:
    """Tensors for the v3 heads. A: pass-A dicts (row = index); B_rows: (row in the pass-B batch, pass-B dict)."""
    ab, ap, gy, ss, se, ob, oa, oy, opos = [], [], [], [], [], [], [], [], []
    Q = max([len(b["q_pos"]) for _, b in B_rows] + [1])
    n_rows = max([r for r, _ in B_rows] + [-1]) + 1
    qpos = torch.full((max(n_rows, 1), Q), -1, dtype=torch.long)
    for r, b in B_rows:
        qpos[r, : len(b["q_pos"])] = torch.tensor(b["q_pos"], dtype=torch.long)
        q0 = b["q_pos"][0] if b["q_pos"] else 0
        clean = set(b.get("_clean", b["anchors"]))     # set by data._mask_slots for training batches
        for anchor, ct in zip(b["anchors"], b.get("copy_targets", [None] * len(b["anchors"]))):
            if anchor not in clean:
                continue
            ab.append(r); ap.append(anchor)
            gy.append(1.0 if ct else 0.0)
            ss.append(ct[0] - q0 if ct else -100); se.append(ct[1] - q0 if ct else -100)
        for (_, _, anchor, opts, tgt) in b["opt_slots"]:
            ob.append(r); oa.append(anchor); oy.append(tgt); opos.append(opts)
    M = max([len(o) for o in opos] + [1])
    opt_pos = torch.full((len(opos), M), -1, dtype=torch.long)
    for n, o in enumerate(opos):
        opt_pos[n, : len(o)] = torch.tensor(o, dtype=torch.long)
    cidx, cy = [], []
    for i, a in enumerate(A):
        for (ci, j, k, c) in a.get("count_items", []):
            cidx.append((i, a["call_pos"][ci], a["arg_pos"][(j, k)]))
            cy.append(c)
    L = lambda x, dt=torch.long: torch.tensor(x, dtype=dt)  # noqa: E731
    return {"anchor_b": L(ab), "anchor_p": L(ap), "gate_y": L(gy, torch.float), "span_s": L(ss), "span_e": L(se),
            "qpos": qpos, "opt_b": L(ob), "opt_anchor": L(oa), "opt_pos": opt_pos, "opt_y": L(oy),
            "count_idx": L(cidx).reshape(-1, 3), "count_y": L(cy)}
