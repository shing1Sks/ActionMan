import random

import torch

from actionman.cached import block_masks, encode_context, encode_slots
from actionman.data import FeatureDataset, collate
from actionman.features import build_pass_a, build_pass_b, gold_plan
from actionman.infer import predict
from actionman.model import ActionMan

from test_model import tiny_examples, tiny_model


def sliding_model(tok):
    """Tiny model WITH sliding-window layers (window 4) so the cached path is tested against both layer types."""
    from transformers import ModernBertConfig
    cfg = ModernBertConfig(vocab_size=50368, hidden_size=64, num_hidden_layers=3, num_attention_heads=2,
                           intermediate_size=128, max_position_embeddings=1024, global_attn_every_n_layers=3,
                           local_attention=8, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    return ActionMan.from_encoder(cfg, tok).eval()


def test_cached_path_equals_block_masked_forward(tok):
    model = sliding_model(tok)
    ex = tiny_examples(1)[0]
    a = build_pass_a(tok, ex)
    b = build_pass_b(tok, ex, a, gold_plan(a, ex), with_targets=False, keep_context=True)
    ids = torch.tensor([b["input_ids"]])
    c0 = b["ctx_len"]
    masks = block_masks(torch.ones_like(ids), torch.tensor([c0]), model.mlm.config.sliding_window)
    with torch.no_grad():
        full = model.mlm.model(input_ids=ids, attention_mask=masks).last_hidden_state
        h_ctx, kv = encode_context(model.mlm.model, ids[:, :c0])
        h_slots = encode_slots(model.mlm.model, ids[:, c0:], c0, kv)
    assert torch.allclose(full[:, :c0], h_ctx, atol=1e-4)
    assert torch.allclose(full[:, c0:], h_slots, atol=1e-4)
    # context states with the block mask == plain pass-A forward (context never sees slots)
    with torch.no_grad():
        pa = model.mlm.model(input_ids=torch.tensor([a["input_ids"]])).last_hidden_state
    assert torch.allclose(pa, h_ctx, atol=1e-4)


def test_cached_training_and_prediction(tok):
    model = tiny_model(tok).train()   # all-global tiny model; sliding windows are covered by the equivalence test
    exs = tiny_examples(8)
    ds = FeatureDataset(tok, exs, cached=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(0)
    first = None
    for _ in range(250):
        loss, _ = model.loss(collate([ds[i] for i in range(8)], tok, rng, cached=True))
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first
    hits = sum([(c.name, c.args) for c in predict(model, tok, ex, T=3, cached=True).calls] ==
               [(c.name, c.args) for c in ex.calls] for ex in exs)
    assert hits >= 6, hits
