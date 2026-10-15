"""Datasets, collation (random mask ratio = masked-diffusion training), splits, negatives."""
from __future__ import annotations

import json
import random

import torch

from .features import build_pass_a, build_pass_b, gold_plan
from .schema import Example, example_from_dict, example_to_dict


def load_jsonl(path) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [example_from_dict(json.loads(line)) for line in f if line.strip()]


def save_jsonl(path, examples: list[Example]):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")


def featurize(tok, ex: Example, cached: bool = False, markers: bool = False, v3: bool = False) -> dict | None:
    if v3:
        from .v3feat import build_pass_a_v3, build_pass_b_v3, gold_plan_v3
        a = build_pass_a_v3(tok, ex)
        if a is None:
            return None
        b = None
        if ex.calls and len(ex.calls) <= len(a["call_pos"]):
            b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True, keep_context=cached,
                                markers=markers)
        return {"a": a, "b": b}
    a = build_pass_a(tok, ex)
    if a is None:
        return None
    b = None
    if ex.calls and len(ex.calls) <= len(a["call_pos"]):
        b = build_pass_b(tok, ex, a, gold_plan(a, ex), with_targets=True, keep_context=cached, markers=markers)
    return {"a": a, "b": b}


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, tok, examples: list[Example], cached: bool = False, markers: bool = False, v3: bool = False):
        self.tok, self.examples, self.cached, self.markers, self.v3 = tok, examples, cached, markers, v3

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return featurize(self.tok, self.examples[i], self.cached, self.markers, self.v3)

    def length(self, i) -> int:
        """Cheap length proxy for bucketing (chars/3)."""
        ex = self.examples[i]
        n = len(ex.query) + sum(len(t.name) + len(t.description) + 20 * len(t.params) for t in ex.tools)
        return n // 3 + 40


def _pad(seqs, value):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), value, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    return ids, attn


def _mask_slots(b, rng, v3_mask: str = "clean"):
    """Masked-diffusion corruption: random ratio r ~ U(0,1], mask that share of slot tokens, reveal the rest.
    v3_mask="legacy" reproduces the original v3 run (s10): every anchor always masked, gate trained on all anchors."""
    ids = list(b["input_ids"])
    flat = [(p, t) for pos, tg in zip(b["slot_pos"], b["slot_targets"]) for p, t in zip(pos, tg)]
    if not flat:                                   # v3: option-only slot block
        return ids, []
    n_mask = max(1, round(rng.uniform(0.0, 1.0) * len(flat)))
    masked = set(rng.sample(range(len(flat)), n_mask))
    always = set()
    if b.get("anchors") and v3_mask == "legacy":
        always = set(b["anchors"])
        b.pop("_clean", None)
    elif b.get("anchors"):
        # v3: half the values are fully masked ("clean"); only their anchors train the copy gate / span pointer,
        # matching inference, where the gate is read once with every slot masked. Other values follow the random
        # mask, so revealed anchors (as in later refinement passes) are seen too.
        clean = [pos for pos in b["slot_pos"] if rng.random() < 0.5]
        always = {p for pos in clean for p in pos}
        b["_clean"] = [pos[0] for pos in clean]
    targets = []
    for idx, (p, t) in enumerate(flat):
        if idx in masked or p in always:
            targets.append((p, t))
        else:
            ids[p] = t
    return ids, targets


def collate(items, tok, rng: random.Random | None = None, cached: bool = False, v3_mask: str = "clean"):
    rng = rng or random
    items = [it for it in items if it is not None]
    if not items:
        return None
    if cached:
        return _collate_cached(items, tok, rng, v3_mask)
    A = [it["a"] for it in items]
    a_ids, a_attn = _pad([a["input_ids"] for a in A], tok.pad_token_id)
    J = max(max(len(a["t_pos"]) for a in A), 1)
    t_pos = torch.full((len(A), J), -1, dtype=torch.long)
    for i, a in enumerate(A):
        t_pos[i, : len(a["t_pos"])] = torch.tensor(a["t_pos"], dtype=torch.long)
    dec = torch.tensor([[J if (l == len(a["t_pos"])) else l for l in a["dec_labels"]] for a in A])
    len_idx, len_labels = [], []
    for i, a in enumerate(A):
        for (ci, j, k, b) in a["len_items"]:
            len_idx.append((i, a["call_pos"][ci], a["arg_pos"][(j, k)]))
            len_labels.append(b)
    batch = {"a_ids": a_ids, "a_attn": a_attn, "t_pos": t_pos, "t_mask": t_pos >= 0,
             "call_pos": torch.tensor([a["call_pos"] for a in A]), "dec_labels": dec,
             "len_idx": torch.tensor(len_idx, dtype=torch.long).reshape(-1, 3),
             "len_labels": torch.tensor(len_labels, dtype=torch.long)}

    # pass B with a random mask ratio per example (masked-diffusion objective)
    B = [it["b"] for it in items if it["b"] is not None and (it["b"]["slot_pos"] or it["b"].get("opt_slots"))]
    if B:
        seqs, sb, sp, sl = [], [], [], []
        for bi, b in enumerate(B):
            ids, targets = _mask_slots(b, rng, v3_mask)
            for p, t in targets:
                sb.append(bi); sp.append(p); sl.append(t)
            seqs.append(ids)
        b_ids, b_attn = _pad(seqs, tok.pad_token_id)
        batch.update({"b_ids": b_ids, "b_attn": b_attn, "slot_b": torch.tensor(sb), "slot_p": torch.tensor(sp),
                      "slot_labels": torch.tensor(sl)})
    else:
        batch["b_ids"] = None
    if "q_pos" in A[0]:
        from .v3feat import v3_extras
        batch.update(v3_extras(A, list(enumerate(B))))
    return batch


def _collate_cached(items, tok, rng, v3_mask="clean"):
    """One sequence per example: pass-A context + slot block, read with the block-causal mask."""
    A, seqs, ctx, sb, sp, sl = [], [], [], [], [], []
    for i, it in enumerate(items):
        a, b = it["a"], it["b"]
        A.append(a)
        ctx.append(len(a["input_ids"]))
        if b is not None and (b["slot_pos"] or b.get("opt_slots")):
            ids, targets = _mask_slots(b, rng, v3_mask)
            for p, t in targets:
                sb.append(i); sp.append(p); sl.append(t)
            seqs.append(ids)
        else:
            seqs.append(a["input_ids"])
    ids, attn = _pad(seqs, tok.pad_token_id)
    J = max(max(len(a["t_pos"]) for a in A), 1)
    t_pos = torch.full((len(A), J), -1, dtype=torch.long)
    for i, a in enumerate(A):
        t_pos[i, : len(a["t_pos"])] = torch.tensor(a["t_pos"], dtype=torch.long)
    dec = torch.tensor([[J if (l == len(a["t_pos"])) else l for l in a["dec_labels"]] for a in A])
    len_idx, len_labels = [], []
    for i, a in enumerate(A):
        for (ci, j, k, b) in a["len_items"]:
            len_idx.append((i, a["call_pos"][ci], a["arg_pos"][(j, k)]))
            len_labels.append(b)
    out = {"a_ids": ids, "a_attn": attn, "t_pos": t_pos, "t_mask": t_pos >= 0,
            "call_pos": torch.tensor([a["call_pos"] for a in A]), "dec_labels": dec,
            "len_idx": torch.tensor(len_idx, dtype=torch.long).reshape(-1, 3),
            "len_labels": torch.tensor(len_labels, dtype=torch.long),
            "ctx_len": torch.tensor(ctx), "b_ids": None, "slot_b": torch.tensor(sb, dtype=torch.long),
            "slot_p": torch.tensor(sp, dtype=torch.long), "slot_labels": torch.tensor(sl, dtype=torch.long)}
    if "q_pos" in A[0]:
        from .v3feat import v3_extras
        out.update(v3_extras(A, [(i, it["b"]) for i, it in enumerate(items)
                                 if it["b"] is not None and (it["b"]["slot_pos"] or it["b"].get("opt_slots"))]))
    return out


def token_batches(lengths: list[int], tokens_per_batch: int, rng: random.Random, max_bs: int = 256):
    """Length-bucketed batches (sorted by length, chunked by token budget, shuffled)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > tokens_per_batch or len(cur) >= max_bs):
            batches.append(cur); cur, m = [], lengths[i]
        cur.append(i); cur_max = m
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def hard_negatives(examples: list[Example], n: int, rng: random.Random) -> list[Example]:
    """Same query, same domain: drop every tool the gold answer uses, keep its related neighbours -> no call."""
    out = []
    pool = [e for e in examples if e.calls and len(e.tools) > len({c.name for c in e.calls})]
    rng.shuffle(pool)
    for e in pool[:n]:
        needed = {c.name for c in e.calls}
        tools = [t for t in e.tools if t.name not in needed]
        out.append(Example(e.query, tools, [], id=f"hardneg_{len(out)}", source="xlam_hardneg", gold=[]))
    return out


def irrelevance_negatives(examples: list[Example], n: int, rng: random.Random) -> list[Example]:
    """Pair real queries with tool lists from other examples that don't contain any needed tool."""
    out = []
    pool = [e for e in examples if e.tools]
    while len(out) < n:
        q, other = rng.choice(pool), rng.choice(pool)
        needed = {c.name for c in q.calls}
        tools = [t for t in other.tools if t.name not in needed]
        if not tools or other is q:
            continue
        out.append(Example(q.query, tools, [], id=f"neg_{len(out)}", source="xlam_neg", gold=[]))
    return out
