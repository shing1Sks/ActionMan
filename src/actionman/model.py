"""ActionMan: ModernBERT encoder + MLM head (slot filler) + decision head + length head."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_BUCKETS


class ActionMan(nn.Module):
    def __init__(self, mlm, v3: bool = False):
        super().__init__()
        self.mlm = mlm
        d = mlm.config.hidden_size
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.none = nn.Linear(d, 1)
        self.len_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, N_BUCKETS))
        self.scale = 1.0 / math.sqrt(d)
        self.v3 = v3
        if v3:   # copy switch + span pointer, option picker, list item count
            from .v3feat import N_COUNT
            self.gate = nn.Linear(d, 1)
            self.sq, self.sk, self.eq, self.ek = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
            self.opt_q, self.opt_k = nn.Linear(d, d), nn.Linear(d, d)
            self.count_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, N_COUNT))

    @classmethod
    def from_encoder(cls, name_or_config, tok, attn_impl: str | None = None, v3: bool = False):
        from transformers import ModernBertConfig, ModernBertForMaskedLM
        if isinstance(name_or_config, ModernBertConfig):
            mlm = ModernBertForMaskedLM(name_or_config)
        else:
            kw = {"attn_implementation": attn_impl} if attn_impl else {}
            mlm = ModernBertForMaskedLM.from_pretrained(name_or_config, **kw)
        mlm.resize_token_embeddings(len(tok), pad_to_multiple_of=64, mean_resizing=False)
        dec = mlm.decoder
        if dec.bias is not None and dec.bias.shape[0] != dec.weight.shape[0]:
            bias = torch.zeros(dec.weight.shape[0], dtype=dec.bias.dtype)
            bias[: dec.bias.shape[0]] = dec.bias.data
            dec.bias = nn.Parameter(bias)
            dec.out_features = dec.weight.shape[0]
        return cls(mlm, v3=v3)

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("mlm.")]

    def encode(self, ids, attn):
        return self.mlm.model(input_ids=ids, attention_mask=attn).last_hidden_state

    def decision_logits(self, h, call_pos, t_pos, t_mask):
        bi = torch.arange(h.shape[0], device=h.device)[:, None]
        call_h = h[bi, call_pos]                                               # [B,4,d]
        t_h = h[bi, t_pos.clamp(min=0)]                                        # [B,J,d]
        logits = torch.einsum("bid,bjd->bij", self.q(call_h), self.k(t_h)) * self.scale
        logits = logits.masked_fill(~t_mask[:, None, :], float("-inf"))
        return torch.cat([logits, self.none(call_h)], dim=-1)                 # [B,4,J+1]; NONE = J

    def length_logits(self, h, len_idx):
        """len_idx: [N,3] rows of (batch index, [CALL] position, param-name position)."""
        if not len_idx.numel():
            return h.new_zeros(0, N_BUCKETS)
        hc, ha = h[len_idx[:, 0], len_idx[:, 1]], h[len_idx[:, 0], len_idx[:, 2]]
        return self.len_head(torch.cat([hc, ha, hc * ha], dim=-1))

    def forward_a(self, batch):
        h = self.encode(batch["a_ids"], batch["a_attn"])                     # [B,L,d]
        dec = self.decision_logits(h, batch["call_pos"], batch["t_pos"], batch["t_mask"])
        return dec, self.length_logits(h, batch["len_idx"])

    def forward_b(self, ids, attn, slot_b, slot_p):
        h = self.encode(ids, attn)
        hs = h[slot_b, slot_p]
        return self.mlm.decoder(self.mlm.head(hs))                            # [M,V]

    # ---------- v3 heads ----------

    def copy_logits(self, h, ab, ap, qpos):
        """Anchors (batch ab, pos ap) [N]; qpos [N,Q] query positions (-1 = pad). -> gate [N], start/end [N,Q]."""
        ha = h[ab, ap]
        hq = h[ab[:, None], qpos.clamp(min=0)]
        pad = qpos < 0
        st = torch.einsum("nd,nqd->nq", self.sq(ha), self.sk(hq)) * self.scale
        en = torch.einsum("nd,nqd->nq", self.eq(ha), self.ek(hq)) * self.scale
        return self.gate(ha).squeeze(-1), st.masked_fill(pad, -1e4), en.masked_fill(pad, -1e4)

    def option_logits(self, h, ob, oa, opos):
        """Option anchors (ob, oa) [N]; opos [N,M] positions of their [OPT] markers (-1 = pad) -> [N,M]."""
        ho = h[ob[:, None], opos.clamp(min=0)]
        lg = torch.einsum("nd,nmd->nm", self.opt_q(h[ob, oa]), self.opt_k(ho)) * self.scale
        return lg.masked_fill(opos < 0, -1e4)

    def count_logits(self, h, idx):
        if not idx.numel():
            return h.new_zeros(0, self.count_head[-1].out_features)
        hc, ha = h[idx[:, 0], idx[:, 1]], h[idx[:, 0], idx[:, 2]]
        return self.count_head(torch.cat([hc, ha, hc * ha], dim=-1))

    def v3_losses(self, h_b, h_a, batch):
        z = h_a.sum() * 0
        l_gate = l_span = l_opt = l_cnt = z
        if batch["anchor_b"].numel():
            g, st, en = self.copy_logits(h_b, batch["anchor_b"], batch["anchor_p"], batch["qpos"][batch["anchor_b"]])
            l_gate = F.binary_cross_entropy_with_logits(g.float(), batch["gate_y"])
            sel = batch["span_s"] >= 0
            if sel.any():
                l_span = 0.5 * (F.cross_entropy(st[sel].float(), batch["span_s"][sel]) +
                                F.cross_entropy(en[sel].float(), batch["span_e"][sel]))
        valid = batch["opt_y"] >= 0
        if valid.any():
            lo = self.option_logits(h_b, batch["opt_b"][valid], batch["opt_anchor"][valid], batch["opt_pos"][valid])
            l_opt = F.cross_entropy(lo.float(), batch["opt_y"][valid])
        if batch["count_idx"].numel():
            l_cnt = F.cross_entropy(self.count_logits(h_a, batch["count_idx"]).float(), batch["count_y"])
        parts = {"gate": l_gate.item(), "span": l_span.item(), "opt": l_opt.item(), "count": l_cnt.item()}
        return l_gate + l_span + l_opt + l_cnt, parts

    def loss_v3(self, batch):
        """Standard-mask v3: pass A and pass B are separate forwards."""
        h_a = self.encode(batch["a_ids"], batch["a_attn"])
        dec_logits = self.decision_logits(h_a, batch["call_pos"], batch["t_pos"], batch["t_mask"])
        len_logits = self.length_logits(h_a, batch["len_idx"])
        l_dec = F.cross_entropy(dec_logits.flatten(0, 1).float(), batch["dec_labels"].flatten(), ignore_index=-100)
        l_len = F.cross_entropy(len_logits.float(), batch["len_labels"]) if len_logits.shape[0] else h_a.sum() * 0
        l_slot = h_a.sum() * 0
        h_b = h_a
        if batch.get("b_ids") is not None:
            h_b = self.encode(batch["b_ids"], batch["b_attn"])
            if batch["slot_b"].numel():
                hs = h_b[batch["slot_b"], batch["slot_p"]]
                l_slot = F.cross_entropy(self.mlm.decoder(self.mlm.head(hs)).float(), batch["slot_labels"])
        if batch.get("b_ids") is None:
            batch = {**batch, "anchor_b": batch["anchor_b"][:0], "opt_y": batch["opt_y"][:0]}
        l_v3, p3 = self.v3_losses(h_b, h_a, batch)
        return l_dec + l_len + l_slot + l_v3, {"dec": l_dec.item(), "len": l_len.item(), "slot": l_slot.item(), **p3}

    def loss(self, batch):
        if batch.get("ctx_len") is not None:
            return self.loss_cached(batch)
        if self.v3:
            return self.loss_v3(batch)
        dec_logits, len_logits = self.forward_a(batch)
        l_dec = F.cross_entropy(dec_logits.flatten(0, 1).float(), batch["dec_labels"].flatten(), ignore_index=-100)
        l_len = (F.cross_entropy(len_logits.float(), batch["len_labels"]) if len_logits.shape[0]
                 else dec_logits.sum() * 0)
        if batch.get("b_ids") is not None and batch["slot_b"].numel():
            slot_logits = self.forward_b(batch["b_ids"], batch["b_attn"], batch["slot_b"], batch["slot_p"])
            l_slot = F.cross_entropy(slot_logits.float(), batch["slot_labels"])
        else:
            l_slot = dec_logits.sum() * 0
        total = l_dec + l_len + l_slot
        return total, {"dec": l_dec.item(), "len": l_len.item(), "slot": l_slot.item()}

    def loss_cached(self, batch):
        """Block-causal variant: one forward; heads read context states, which never see the slots."""
        from .cached import block_masks
        masks = block_masks(batch["a_attn"], batch["ctx_len"], self.mlm.config.sliding_window)
        h = self.mlm.model(input_ids=batch["a_ids"], attention_mask=masks).last_hidden_state
        dec_logits = self.decision_logits(h, batch["call_pos"], batch["t_pos"], batch["t_mask"])
        len_logits = self.length_logits(h, batch["len_idx"])
        l_dec = F.cross_entropy(dec_logits.flatten(0, 1).float(), batch["dec_labels"].flatten(), ignore_index=-100)
        l_len = F.cross_entropy(len_logits.float(), batch["len_labels"]) if len_logits.shape[0] else h.sum() * 0
        if batch["slot_b"].numel():
            hs = h[batch["slot_b"], batch["slot_p"]]
            l_slot = F.cross_entropy(self.mlm.decoder(self.mlm.head(hs)).float(), batch["slot_labels"])
        else:
            l_slot = h.sum() * 0
        parts = {"dec": l_dec.item(), "len": l_len.item(), "slot": l_slot.item()}
        total = l_dec + l_len + l_slot
        if self.v3:
            l_v3, p3 = self.v3_losses(h, h, batch)
            total, parts = total + l_v3, {**parts, **p3}
        return total, parts
