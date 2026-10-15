"""Tokenizer setup and the two sequence builders.

Pass A:  [CLS] [T] sig_1 [T] sig_2 ... [USER] query [CALL] x4 [SEP]
Pass B:  <pass A without [SEP]> [ARGS] tool( p= [MASK]*n , q= [MASK]*m ) ... [SEP]
"""
from __future__ import annotations

from .schema import Example, signature_pieces, value_to_text

SPECIAL = ["[T]", "[CALL]", "[ARGS]", "[USER]", "[OPT]"]   # [OPT]: v3 option marker (appended, ids of the rest unchanged)
MAX_CALLS = 4
BUCKETS = [0, 4, 8, 16, 32]          # slot counts for bucket ids 0..4; id 5 = overflow
OVERFLOW = 5
N_BUCKETS = 6
MAX_LEN_A = 896
MAX_LEN_B = 1024


def load_tokenizer(name: str = "answerdotai/ModernBERT-large"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    tok.am = {s: tok.convert_tokens_to_ids(s) for s in SPECIAL}
    return tok


def bucket_of(n_tokens: int) -> int:
    for b in range(1, 5):
        if n_tokens <= BUCKETS[b]:
            return b
    return OVERFLOW


def _enc(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


def value_tokens(tok, value) -> list[int]:
    return _enc(tok, value_to_text(value))


def build_pass_a(tok, ex: Example, desc_words: int = 8, max_len: int = MAX_LEN_A) -> dict | None:
    """Context sequence + decision / length labels. Returns None if it can't fit."""
    for dw in (desc_words, 0):
        ids = [tok.cls_token_id]
        t_pos, arg_pos = [], {}
        for j, tool in enumerate(ex.tools):
            t_pos.append(len(ids))
            ids.append(tok.am["[T]"])
            for text, kind, k in signature_pieces(tool, dw):
                if kind == "param":
                    arg_pos[(j, k)] = len(ids)
                ids += _enc(tok, text)
        ids.append(tok.am["[USER]"])
        ids += _enc(tok, ex.query)
        call_pos = list(range(len(ids), len(ids) + MAX_CALLS))
        ids += [tok.am["[CALL]"]] * MAX_CALLS + [tok.sep_token_id]
        if len(ids) <= max_len:
            break
    else:
        return None

    names = [t.name for t in ex.tools]
    dec_labels, len_items = [], []
    for i in range(MAX_CALLS):
        if i < len(ex.calls):
            c = ex.calls[i]
            if c.name not in names:
                return None
            j = names.index(c.name)
            dec_labels.append(j)
            for k, p in enumerate(ex.tools[j].params):
                b = bucket_of(len(value_tokens(tok, c.args[p.name]))) if p.name in c.args else 0
                len_items.append((i, j, k, b))
        else:
            dec_labels.append(len(ex.tools))  # NONE
    if len(ex.calls) > MAX_CALLS:
        dec_labels = [-100] * MAX_CALLS     # can't represent; train nothing on decisions
        len_items = [it for it in len_items if it[0] < MAX_CALLS]
    return {"input_ids": ids, "t_pos": t_pos, "call_pos": call_pos, "arg_pos": arg_pos,
            "dec_labels": dec_labels, "len_items": len_items, "desc_words": dw}


def build_pass_b(tok, ex: Example, a: dict, plan: list[tuple[int, dict[int, int]]],
                 with_targets: bool, max_len: int = MAX_LEN_B, keep_context: bool = False,
                 markers: bool = False) -> dict | None:
    """Append the slot block for `plan` = [(tool_j, {param_k: bucket})] per call.
    Buckets 1..4 get BUCKETS[b] mask slots; 0 (absent) and 5 (overflow) get none."""
    # keep_context=True (cached variant): the full pass-A sequence is the prefix, so its states can be reused
    ids = (a["input_ids"] if keep_context else a["input_ids"][:-1]) + [tok.am["[ARGS]"]]
    ctx_len = len(a["input_ids"]) if keep_context else len(ids) - 1
    slots, targets, index = [], [], []
    for i, (j, buckets) in enumerate(plan):
        tool = ex.tools[j]
        # markers=True (v2): '#1 tool(' '#2 tool(' so repeated calls of the same tool are distinguishable
        ids += _enc(tok, (f" #{i + 1} " if markers else "") + tool.name + "(")
        first = True
        for k, p in enumerate(tool.params):
            b = buckets.get(k, 0)
            if b == 0 or b == OVERFLOW:
                continue
            ids += _enc(tok, ("" if first else ", ") + p.name + "=")
            first = False
            n = BUCKETS[b]
            pos = list(range(len(ids), len(ids) + n))
            ids += [tok.mask_token_id] * n
            slots.append(pos)
            index.append((i, k))
            if with_targets:
                v = value_tokens(tok, ex.calls[i].args[p.name])[:n]
                targets.append(v + [tok.pad_token_id] * (n - len(v)))
        ids += _enc(tok, ")")
    ids.append(tok.sep_token_id)
    if len(ids) > max_len:
        return None
    out = {"input_ids": ids, "slot_pos": slots, "slot_index": index, "ctx_len": ctx_len}
    if with_targets:
        out["slot_targets"] = targets
    return out


def gold_plan(a: dict, ex: Example) -> list[tuple[int, dict[int, int]]]:
    """One (tool_j, {param_k: bucket}) entry per gold call, in call order - including calls to tools
    with no parameters (an empty dict), so plan index == call index."""
    plan = []
    for i in range(min(len(ex.calls), MAX_CALLS)):
        buckets = {k: b for (ci, _, k, b) in a["len_items"] if ci == i}
        plan.append((a["dec_labels"][i], buckets))
    return plan
