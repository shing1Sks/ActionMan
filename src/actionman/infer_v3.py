"""v3 inference: pass A (tools, lengths, list counts) -> slot block with option / item slots ->
copy-or-generate per value (decided on the first pass) -> mask-predict refinement of the generated slots."""
from __future__ import annotations

from collections import defaultdict

import torch

from .features import MAX_CALLS, OVERFLOW
from .infer import Prediction, PredCall, _banned_ids
from .schema import Example, param_options, text_to_value
from .span import best_span, snap_word
from .v3feat import build_pass_a_v3, build_pass_b_v3


@torch.no_grad()
def predict_v3(model, tok, ex: Example, T: int = 2, tau: float = 0.85, device="cpu", cached: bool = False,
               markers: bool = True, snap: bool = False, require: bool = False) -> Prediction:
    """snap: widen copied spans to whole words/numbers (inference-only fix for copies cut mid-word).
    require: like tool_choice="required" -- the first call never abstains; it takes the best tool instead."""
    model.eval()
    ex0 = Example(ex.query, ex.tools, [], ex.id, ex.source)
    a = build_pass_a_v3(tok, ex0)
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
    chosen = []
    for i in range(MAX_CALLS):
        j = int(probs[i].argmax())
        if j == J and require and i == 0:
            j = int(probs[i, :J].argmax())
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
    idx = torch.tensor(rows, device=device).reshape(-1, 3)
    len_logits = model.length_logits(h, idx).float()
    cnt_logits = model.count_logits(h, idx).float()
    plan, preds = [], []
    for i, j, conf in chosen:
        modes, pc = {}, PredCall(ex.tools[j].name, conf_tool=conf)
        for r, (ri, _, k) in enumerate(keys):
            if ri != i:
                continue
            p = ex.tools[j].params[k]
            lg = len_logits[r].clone()
            if not p.optional:
                lg[0] = float("-inf")
            b = int(lg.argmax())
            if b == 0:
                continue
            if param_options(p) is not None:
                modes[k] = ("opt", 0)
            elif p.type == "list" and int(cnt_logits[r].argmax()) >= 1:
                modes[k] = ("list", int(cnt_logits[r].argmax()))
            else:
                modes[k] = ("text", b)
                if b == OVERFLOW:
                    pc.escalated.append(p.name)
        plan.append((j, modes))
        preds.append(pc)

    bb = build_pass_b_v3(tok, ex0, a, plan, with_targets=False, keep_context=cached, markers=markers)
    if bb is None or not (bb["slot_pos"] or bb["opt_slots"]):
        return Prediction(preds, 2)
    ids_b = torch.tensor([bb["input_ids"]], device=device)
    c0 = bb["ctx_len"]

    def forward():
        if cached:
            hs = encode_slots(model.mlm.model, ids_b[:, c0:], c0, kv)
            return torch.cat([h, hs], dim=1)
        return model.encode(ids_b, torch.ones_like(ids_b))

    flat = [p for pos in bb["slot_pos"] for p in pos]
    fpos = torch.tensor(flat, dtype=torch.long, device=device)
    conf = torch.zeros(len(flat), device=device)
    masked = torch.ones(len(flat), dtype=torch.bool, device=device)
    fixed = torch.zeros(len(flat), dtype=torch.bool, device=device)
    vocab = model.mlm.decoder.weight.shape[0]
    banned = _banned_ids(tok, vocab).to(device)
    copied = {}                                     # value index -> (text, conf)
    starts = [0]
    for pos in bb["slot_pos"]:
        starts.append(starts[-1] + len(pos))
    passes, hb = 0, None
    for t in range(T):
        passes = t + 1
        hb = forward()
        if t == 0 and bb["anchors"]:
            ab = torch.zeros(len(bb["anchors"]), dtype=torch.long, device=device)
            ap = torch.tensor(bb["anchors"], device=device)
            qpos = torch.tensor([a["q_pos"]], device=device).expand(len(bb["anchors"]), -1)
            g, st, en = model.copy_logits(hb, ab, ap, qpos)
            g = g.float().sigmoid()
            for v in range(len(bb["anchors"])):
                if g[v] <= 0.5 or not a["q_pos"]:
                    continue
                s, e, pspan = best_span(st[v], en[v])
                cs, ce = a["q_offs"][s][0], a["q_offs"][e][1]
                if snap:
                    cs, ce = snap_word(ex.query, cs, ce)
                text = ex.query[cs: ce].strip()
                if not text:
                    continue
                copied[v] = (text, float(g[v]) * pspan)
                n = len(bb["slot_pos"][v])
                span_ids = [a["input_ids"][q] for q in a["q_pos"][s: e + 1]][:n]
                span_ids += [tok.pad_token_id] * (n - len(span_ids))
                sl = slice(starts[v], starts[v + 1])
                ids_b[0, fpos[sl]] = torch.tensor(span_ids, device=device)
                fixed[sl] = True
        if not flat:
            break
        logits = model.mlm.decoder(model.mlm.head(hb[0, fpos])).float()
        logits[:, banned] = float("-inf")
        p, tokens = logits.softmax(-1).max(-1)
        upd = masked & ~fixed
        conf = torch.where(upd, p, conf)
        ids_b[0, fpos[upd]] = tokens[upd]
        conf = torch.where(fixed, torch.ones_like(conf), conf)
        if t == T - 1:
            break
        masked = (conf < tau) & ~fixed
        if not masked.any():
            break
        ids_b[0, fpos[masked]] = tok.mask_token_id

    ci_of = {c[0]: n for n, c in enumerate(chosen)}
    out_tokens, confs = ids_b[0, fpos].tolist() if flat else [], conf.tolist()
    items = defaultdict(list)                       # (call, param) -> [(text, conf)] for list items
    for v, ((i, k, item), pos) in enumerate(zip(bb["slot_index"], bb["slot_pos"])):
        if v in copied:
            text, cv = copied[v]
        else:
            toks = out_tokens[starts[v]: starts[v + 1]]
            if tok.pad_token_id in toks:
                toks = toks[: toks.index(tok.pad_token_id)]
            text = tok.decode(toks, skip_special_tokens=True).strip()
            cs = confs[starts[v]: starts[v + 1]]
            cv = float(min(cs)) if cs else 1.0
        ci = ci_of[i]
        param = ex.tools[chosen[ci][1]].params[k]
        if item is None:
            preds[ci].args[param.name] = text_to_value(text, param.type)
            preds[ci].conf_args[param.name] = cv
        else:
            items[(ci, param.name)].append((text, cv))
    for (ci, name), vals in items.items():
        preds[ci].args[name] = [text_to_value(t, "any") for t, _ in vals]
        preds[ci].conf_args[name] = min(c for _, c in vals)
    if bb["opt_slots"]:
        ob = torch.zeros(len(bb["opt_slots"]), dtype=torch.long, device=device)
        oa = torch.tensor([s[2] for s in bb["opt_slots"]], device=device)
        M = max(len(s[3]) for s in bb["opt_slots"])
        opos = torch.full((len(bb["opt_slots"]), M), -1, dtype=torch.long, device=device)
        for n, s in enumerate(bb["opt_slots"]):
            opos[n, : len(s[3])] = torch.tensor(s[3], device=device)
        po = model.option_logits(hb, ob, oa, opos).float().softmax(-1)
        for n, (i, k, _, _, _) in enumerate(bb["opt_slots"]):
            ci = ci_of[i]
            param = ex.tools[chosen[ci][1]].params[k]
            m = int(po[n].argmax())
            preds[ci].args[param.name] = param_options(param)[m]
            preds[ci].conf_args[param.name] = float(po[n, m])
    return Prediction(preds, passes + 1)
