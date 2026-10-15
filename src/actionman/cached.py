"""Block-causal 'cached' variant (Stage 4).

Sequence = [pass-A context incl. [CALL]x4 [SEP]] [ARGS] slot block [SEP].
Mask rule: context tokens see only context; slot-block tokens see everything.
Consequences:
  * training needs ONE forward (decision/length heads read context states that can't see the slots);
  * at inference the context K/V of every layer is computed once (that IS pass A) and each refinement
    pass runs only the ~10-40 slot tokens through the network.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


def block_masks(attn: torch.Tensor, ctx_len: torch.Tensor, window: int) -> dict:
    """attn [B,L] (1 = real token), ctx_len [B]. Returns HF ModernBERT mask dict (bool, True = attend)."""
    B, L = attn.shape
    idx = torch.arange(L, device=attn.device)
    valid_k = attn.bool()[:, None, None, :]
    q_slot = (idx[None, :] >= ctx_len[:, None])[:, None, :, None]
    k_ctx = (idx[None, :] < ctx_len[:, None])[:, None, None, :]
    eye = torch.eye(L, dtype=torch.bool, device=attn.device)[None, None]
    full = (valid_k & (q_slot | k_ctx)) | eye
    win = ((idx[:, None] - idx[None, :]).abs() <= window)[None, None]
    return {"full_attention": full, "sliding_attention": full & win}


def _layer_qkv(layer, x, cos, sin):
    attn = layer.attn
    B, L, _ = x.shape
    qkv = attn.Wqkv(layer.attn_norm(x)).view(B, L, 3, -1, attn.head_dim)
    q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    return q, k, v


def _finish(layer, x, out):
    B, L = x.shape[:2]
    x = x + layer.attn.Wo(out.transpose(1, 2).reshape(B, L, -1))
    return x + layer.mlp(layer.mlp_norm(x))


@torch.no_grad()
def encode_context(enc, ids: torch.Tensor):
    """enc = ModernBertModel. ids [1,Lc]. Returns (final hidden [1,Lc,d], per-layer (k, v) cache)."""
    cfg = enc.config
    x = enc.embeddings(input_ids=ids)
    L = ids.shape[1]
    pos = torch.arange(L, device=ids.device)[None]
    pe = {lt: enc.rotary_emb(x, pos, lt) for lt in set(cfg.layer_types)}
    idx = torch.arange(L, device=ids.device)
    win = ((idx[:, None] - idx[None, :]).abs() <= cfg.sliding_window)[None, None]
    cache = []
    for layer in enc.layers:
        cos, sin = pe[layer.attention_type]
        q, k, v = _layer_qkv(layer, x, cos, sin)
        mask = win if layer.attention_type == "sliding_attention" else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=layer.attn.head_dim ** -0.5)
        x = _finish(layer, x, out)
        cache.append((k, v))
    return enc.final_norm(x), cache


@torch.no_grad()
def encode_slots(enc, slot_ids: torch.Tensor, start: int, cache):
    """Run only the slot-block tokens (absolute positions start..start+S-1) over the cached context K/V."""
    cfg = enc.config
    x = enc.embeddings(input_ids=slot_ids)
    S = slot_ids.shape[1]
    Lc = cache[0][0].shape[2]
    qpos = torch.arange(start, start + S, device=slot_ids.device)
    kpos = torch.cat([torch.arange(Lc, device=slot_ids.device), qpos])
    pe = {lt: enc.rotary_emb(x, qpos[None], lt) for lt in set(cfg.layer_types)}
    win = ((qpos[:, None] - kpos[None, :]).abs() <= cfg.sliding_window)[None, None]
    for layer, (ck, cv) in zip(enc.layers, cache):
        cos, sin = pe[layer.attention_type]
        q, k, v = _layer_qkv(layer, x, cos, sin)
        k, v = torch.cat([ck, k], dim=2), torch.cat([cv, v], dim=2)
        mask = win if layer.attention_type == "sliding_attention" else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=layer.attn.head_dim ** -0.5)
        x = _finish(layer, x, out)
    return enc.final_norm(x)


# ---------- static-shape variants for CUDA graphs / torch.compile (v3 S6) ----------
# Lengths are padded up to a bucket and the real lengths are passed as tensors, so one compiled graph serves every
# prompt in a bucket. Padded keys are masked out; outputs at padded positions are garbage and get sliced off.

def _ctx_core(enc, ids, valid):
    """ids [1,P] padded; valid = tensor([real length]). -> hidden [1,P,d], cache."""
    cfg = enc.config
    x = enc.embeddings(input_ids=ids)
    P = ids.shape[1]
    idx = torch.arange(P, device=ids.device)
    pos = idx[None]
    pe = {lt: enc.rotary_emb(x, pos, lt) for lt in set(cfg.layer_types)}
    keyok = (idx < valid)[None, None, None, :]
    eye = torch.eye(P, dtype=torch.bool, device=ids.device)[None, None]
    full = keyok | eye
    win = full & ((idx[:, None] - idx[None, :]).abs() <= cfg.sliding_window)[None, None]
    cache = []
    for layer in enc.layers:
        cos, sin = pe[layer.attention_type]
        q, k, v = _layer_qkv(layer, x, cos, sin)
        mask = win if layer.attention_type == "sliding_attention" else full
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=layer.attn.head_dim ** -0.5)
        x = _finish(layer, x, out)
        cache.append((k, v))
    return enc.final_norm(x), cache


def _slot_core(enc, slot_ids, start, s_valid, c_valid, cache):
    """slot_ids [1,Sp] padded; start/s_valid/c_valid tensors; cache from _ctx_core (padded context)."""
    cfg = enc.config
    x = enc.embeddings(input_ids=slot_ids)
    Sp = slot_ids.shape[1]
    Lc = cache[0][0].shape[2]
    ar = torch.arange(Sp, device=slot_ids.device)
    qpos = start + ar
    kc = torch.arange(Lc, device=slot_ids.device)
    kpos = torch.cat([kc, qpos])
    keyok = torch.cat([kc < c_valid, ar < s_valid])[None, None, None, :]
    pe = {lt: enc.rotary_emb(x, qpos[None], lt) for lt in set(cfg.layer_types)}
    full = keyok | torch.cat([torch.zeros(Sp, Lc, dtype=torch.bool, device=slot_ids.device),
                              torch.eye(Sp, dtype=torch.bool, device=slot_ids.device)], dim=1)[None, None]
    win = full & ((qpos[:, None] - kpos[None, :]).abs() <= cfg.sliding_window)[None, None]
    for layer, (ck, cv) in zip(enc.layers, cache):
        cos, sin = pe[layer.attention_type]
        q, k, v = _layer_qkv(layer, x, cos, sin)
        k, v = torch.cat([ck, k], dim=2), torch.cat([cv, v], dim=2)
        mask = win if layer.attention_type == "sliding_attention" else full
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=layer.attn.head_dim ** -0.5)
        x = _finish(layer, x, out)
    return enc.final_norm(x)


def _bucket(n: int, b: int) -> int:
    return -(-n // b) * b


def make_static(enc, ctx_bucket: int = 256, slot_bucket: int = 128, compile: bool = True):
    """-> (encode_context, encode_slots) drop-ins that pad to buckets (and run compiled, CUDA-graphed cores).
    Buckets are coarse on purpose: every (context bucket, slot bucket) pair is a separate compiled graph."""
    ctx_core, slot_core = _ctx_core, _slot_core
    if compile:
        import torch._dynamo as dynamo
        for name in ("cache_size_limit", "recompile_limit"):
            if hasattr(dynamo.config, name):
                setattr(dynamo.config, name, 64)
        ctx_core = torch.compile(_ctx_core, mode="reduce-overhead", dynamic=False, fullgraph=False)
        slot_core = torch.compile(_slot_core, mode="reduce-overhead", dynamic=False, fullgraph=False)
    pad_id = enc.config.pad_token_id or 0
    state = {}

    @torch.no_grad()
    def encode_context_s(enc_, ids):
        L = ids.shape[1]
        P = _bucket(L, ctx_bucket)
        ids_p = F.pad(ids, (0, P - L), value=pad_id)
        h, cache = ctx_core(enc, ids_p, torch.tensor(L, device=ids.device))
        h, cache = h.clone(), [(k.clone(), v.clone()) for k, v in cache]   # outlive the CUDA-graph buffers
        state["c_valid"] = L
        return h[:, :L], cache

    @torch.no_grad()
    def encode_slots_s(enc_, slot_ids, start, cache):
        S = slot_ids.shape[1]
        Sp = _bucket(S, slot_bucket)
        ids_p = F.pad(slot_ids, (0, Sp - S), value=pad_id)
        dev = slot_ids.device
        out = slot_core(enc, ids_p, torch.tensor(start, device=dev), torch.tensor(S, device=dev),
                        torch.tensor(state["c_valid"], device=dev), cache)
        return out[:, :S].clone()

    return encode_context_s, encode_slots_s
