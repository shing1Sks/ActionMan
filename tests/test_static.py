import torch

from actionman.cached import encode_context, encode_slots, make_static

from test_v3 import tiny_v3


def test_padded_static_encoders_match_eager(tok):
    model = tiny_v3(tok, global_every=2)                   # mix of global and sliding-window layers
    model.mlm.config.sliding_window = 4
    enc = model.mlm.model.eval()
    ctx = torch.randint(5, 1000, (1, 37))
    slots = torch.randint(5, 1000, (1, 11))
    h0, kv0 = encode_context(enc, ctx)
    s0 = encode_slots(enc, slots, 37, kv0)
    ec, es = make_static(enc, ctx_bucket=16, slot_bucket=8, compile=False)
    h1, kv1 = ec(enc, ctx)
    s1 = es(enc, slots, 37, kv1)
    assert torch.allclose(h0, h1, atol=1e-5) and torch.allclose(s0, s1, atol=1e-5)
