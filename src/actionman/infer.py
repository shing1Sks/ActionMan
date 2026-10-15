"""Inference: pass A (decide tools + arg lengths) -> slot block -> mask-predict refinement."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .features import BUCKETS, MAX_CALLS, OVERFLOW, SPECIAL, build_pass_a, build_pass_b
from .schema import Call, Example, text_to_value


@dataclass
class PredCall:
    name: str
    args: dict = field(default_factory=dict)
    conf_tool: float = 0.0
    conf_args: dict = field(default_factory=dict)
    escalated: list = field(default_factory=list)

    def as_call(self) -> Call:
        return Call(self.name, self.args)


@dataclass
class Prediction:
    calls: list[PredCall]
    passes: int = 0
    skipped: bool = False


def _banned_ids(tok, vocab: int) -> torch.Tensor:
    ids = [tok.mask_token_id, tok.cls_token_id, tok.sep_token_id] + [tok.am[s] for s in SPECIAL]
    ids += list(range(len(tok), vocab))
    return torch.tensor(sorted(set(i for i in ids if i < vocab)))


@torch.no_grad()
def predict(model, tok, ex: Example, T: int = 2, tau: float = 0.85, device="cpu", cached: bool = False,
            markers: bool = False, v3: bool = False, snap: bool = False, require: bool = False) -> Prediction:
    if v3:
        from .infer_v3 import predict_v3
        return predict_v3(model, tok, ex, T=T, tau=tau, device=device, cached=cached, markers=markers, snap=snap, require=require)
    model.eval()
    ex0 = Example(ex.query, ex.tools, [], ex.id, ex.source)
    a = build_pass_a(tok, ex0)
    if a is None or not ex.tools:
        return Prediction([], 0, skipped=a is None)
    ids = torch.tensor([a["input_ids"]], device=device)
    if cached:
        from .cached import encode_context, encode_slots
        h, kv = encode_context(model.mlm.model, ids)
    else:
        h = model.encode(ids, torch.ones_like(ids))
    J = len(a["t_pos"])
    call_pos = torch.tensor([a["call_pos"]], device=device)
    t_pos = torch.tensor([a["t_pos"]], device=device)
    probs = model.decision_logits(h, call_pos, t_pos, torch.ones_like(t_pos, dtype=torch.bool)).float().softmax(-1)[0]

    chosen = []                                   # (call_i, tool_j, conf)
    for i in range(MAX_CALLS):
        j = int(probs[i].argmax())
        if j == J:
            break
        chosen.append((i, j, float(probs[i, j])))
    if not chosen:
        return Prediction([], 1)

    rows, keys = [], []
    for i, j, _ in chosen:
        for k in range(len(ex.tools[j].params)):
            rows.append((0, a["call_pos"][i], a["arg_pos"][(j, k)]))
            keys.append((i, j, k))
    len_logits = model.length_logits(h, torch.tensor(rows, device=device).reshape(-1, 3)).float()
    plan, preds = [], []
    for ci, (i, j, conf) in enumerate(chosen):
        buckets, pc = {}, PredCall(ex.tools[j].name, conf_tool=conf)
        for r, (ri, rj, k) in enumerate(keys):
            if ri != i:
                continue
            lg = len_logits[r].clone()
            if not ex.tools[j].params[k].optional:
                lg[0] = float("-inf")            # required params can't be absent
            b = int(lg.argmax())
            buckets[k] = b
            if b == OVERFLOW:
                pc.escalated.append(ex.tools[j].params[k].name)
        plan.append((j, buckets))
        preds.append(pc)

    b = build_pass_b(tok, ex0, a, plan, with_targets=False, keep_context=cached, markers=markers)
    passes = 1
    if b is not None and b["slot_pos"]:
        ids_b = torch.tensor([b["input_ids"]], device=device)
        flat = [p for pos in b["slot_pos"] for p in pos]
        fpos = torch.tensor(flat, device=device)
        conf = torch.zeros(len(flat), device=device)
        masked = torch.ones(len(flat), dtype=torch.bool, device=device)
        vocab = model.mlm.decoder.weight.shape[0]
        banned = _banned_ids(tok, vocab).to(device)
        for t in range(T):
            passes = t + 1
            if cached:
                c0 = b["ctx_len"]
                hs = encode_slots(model.mlm.model, ids_b[:, c0:], c0, kv)[0, fpos - c0]
                logits = model.mlm.decoder(model.mlm.head(hs)).float()
            else:
                logits = model.forward_b(ids_b, torch.ones_like(ids_b), torch.zeros_like(fpos), fpos).float()
            logits[:, banned] = float("-inf")
            p, tokens = logits.softmax(-1).max(-1)
            conf = torch.where(masked, p, conf)
            ids_b[0, fpos[masked]] = tokens[masked]
            if t == T - 1:
                break
            masked = conf < tau
            if not masked.any():
                break
            ids_b[0, fpos[masked]] = tok.mask_token_id
        out_tokens = ids_b[0, fpos].tolist()
        confs = conf.tolist()
        cursor = 0
        for (i, k), pos in zip(b["slot_index"], b["slot_pos"]):
            n = len(pos)
            toks, cs = out_tokens[cursor: cursor + n], confs[cursor: cursor + n]
            cursor += n
            if tok.pad_token_id in toks:
                toks = toks[: toks.index(tok.pad_token_id)]
            text = tok.decode(toks, skip_special_tokens=True).strip()
            ci = [c[0] for c in chosen].index(i)
            param = ex.tools[chosen[ci][1]].params[k]
            preds[ci].args[param.name] = text_to_value(text, param.type)
            preds[ci].conf_args[param.name] = float(min(cs)) if cs else 1.0
    return Prediction(preds, passes + 1)
