import random

import torch

from actionman.data import FeatureDataset, collate
from actionman.model import ActionMan

from conftest import make_example


def tiny_model(tok):
    from transformers import ModernBertConfig
    cfg = ModernBertConfig(vocab_size=50368, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=128, max_position_embeddings=1024, global_attn_every_n_layers=1,
                           pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    return ActionMan.from_encoder(cfg, tok)


def tiny_examples(n):
    qs = ["best ramen near shibuya", "cheap flights to goa", "weather pune", "python list sort",
          "nearest atm", "news today", "football scores", "movie times"]
    return [make_example(query=f"find {qs[i % len(qs)]} top {i % 5 + 1}", q=qs[i % len(qs)], k=i % 5 + 1)
            for i in range(n)]


def test_tiny_model_learns(tok):
    model = tiny_model(tok)
    ds = FeatureDataset(tok, tiny_examples(16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(0)
    first = None
    for step in range(60):
        batch = collate([ds[i] for i in range(16)], tok, rng)
        loss, parts = model.loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first, (first, loss.item(), parts)
