import random

import torch

from actionman.data import FeatureDataset, collate
from actionman.infer import predict
from actionman.schema import Example

from test_model import tiny_examples, tiny_model


def test_overfit_then_predict(tok):
    model = tiny_model(tok)
    exs = tiny_examples(8)
    ds = FeatureDataset(tok, exs)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(0)
    for _ in range(250):
        batch = collate([ds[i] for i in range(8)], tok, rng)
        loss, _ = model.loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    hits = 0
    for ex in exs:
        pred = predict(model, tok, ex, T=3)
        got = [(c.name, c.args) for c in pred.calls]
        hits += got == [(c.name, c.args) for c in ex.calls]
    assert hits >= 6, hits


def test_predict_never_raises_on_empty_tools(tok):
    model = tiny_model(tok)
    pred = predict(model, tok, Example("hello", [], [], "x", "t"))
    assert pred.calls == []
