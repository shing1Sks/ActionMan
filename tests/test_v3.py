import random

import pytest
import torch

from actionman.data import FeatureDataset, collate
from actionman.infer import predict
from actionman.model import ActionMan
from actionman.schema import Call, Example, Param, Tool
from actionman.span import best_span
from actionman.v3feat import build_pass_a_v3, build_pass_b_v3, gold_plan_v3


def tiny_v3(tok, global_every=1):
    from transformers import ModernBertConfig
    cfg = ModernBertConfig(vocab_size=50368, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=128, max_position_embeddings=1024, global_attn_every_n_layers=global_every,
                           pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    return ActionMan.from_encoder(cfg, tok, v3=True)


def ex_copy():
    t = Tool("find_food", "find restaurants", [Param("city", "str"), Param("cuisine", "str", True)])
    return Example("Find sushi near Boston tonight", [t], [Call("find_food", {"city": "Boston", "cuisine": "sushi"})])


def ex_opt(gold="imperial"):
    t = Tool("bmi", "body mass index", [Param("weight", "int"),
                                        Param("unit", "str", False, None, "", ["metric", "imperial"]),
                                        Param("verbose", "bool", True)])
    return Example("bmi for 200 lbs in imperial please", [t], [Call("bmi", {"weight": 200, "unit": gold, "verbose": True})])


def ex_list(items=("Apple", "Tesla")):
    t = Tool("prices", "stock prices", [Param("companies", "list")])
    return Example(f"prices for {' and '.join(items)}", [t], [Call("prices", {"companies": list(items)})])


def test_best_span_respects_order_and_max_len():
    st = torch.tensor([0.0, 5.0, 0.0, 0.0])
    en = torch.tensor([9.0, 0.0, 0.0, 6.0])
    s, e, p = best_span(st, en, max_len=32)
    assert (s, e) == (1, 3) and 0 < p <= 1                 # end=0 lies before start=1, so it is not allowed
    s, e, _ = best_span(st, en, max_len=1)
    assert s == e                                          # max_len=1 -> single-token spans only


def test_pass_a_v3_query_positions(tok):
    ex = ex_copy()
    a = build_pass_a_v3(tok, ex)
    assert tok.decode([a["input_ids"][p] for p in a["q_pos"]]).strip() == ex.query


def test_pass_b_v3_copy_target_points_at_query(tok):
    ex = ex_copy()
    a = build_pass_a_v3(tok, ex)
    b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True)
    s, e = b["copy_targets"][0]
    assert tok.decode(b["input_ids"][s: e + 1]).strip().lower() == "boston"
    assert b["anchors"] == [p[0] for p in b["slot_pos"]]


def test_option_block_layout_and_target(tok):
    ex = ex_opt()
    a = build_pass_a_v3(tok, ex)
    b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True)
    (i1, k1, anc1, opts1, t1), (i2, k2, anc2, opts2, t2) = b["opt_slots"]
    assert all(b["input_ids"][p] == tok.am["[OPT]"] for p in opts1 + opts2)
    assert (k1, t1, k2, t2) == (1, 1, 2, 0)              # unit -> imperial (1); verbose bool -> True (0)
    assert b["input_ids"][anc1] == tok.mask_token_id


def test_option_gold_not_in_options_falls_back_to_generate(tok):
    ex = ex_opt("kelvin")
    a = build_pass_a_v3(tok, ex)
    b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True)
    assert [s[1] for s in b["opt_slots"]] == [2]          # only the bool is an option slot
    assert [ix[1] for ix in b["slot_index"]] == [0, 1]    # weight + unit are generated


def test_list_items_get_own_slots_and_count_label(tok):
    ex = ex_list()
    a = build_pass_a_v3(tok, ex)
    assert a["count_items"] == [(0, 0, 0, 2)]
    b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True)
    assert [ix[2] for ix in b["slot_index"]] == [0, 1] and all(len(p) == 8 for p in b["slot_pos"])
    assert all(ct is not None for ct in b["copy_targets"])


def test_long_list_uses_text_path(tok):
    ex = ex_list(tuple("abcdefgh"))                     # 8 items (> MAX_ITEMS) but short as text
    a = build_pass_a_v3(tok, ex)
    assert a["count_items"] == [(0, 0, 0, 0)]
    b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True)
    assert [ix[2] for ix in b["slot_index"]] == [None]


def test_zero_param_and_option_only_calls_stay_aligned(tok):
    ping = Tool("ping", "ping", [])
    flag = Tool("flag", "set flag", [Param("on", "bool")])
    ex = Example("ping then flag on, then ping", [ping, flag],
                 [Call("ping", {}), Call("flag", {"on": True}), Call("ping", {})])
    a = build_pass_a_v3(tok, ex)
    b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True)
    assert [(s[0], s[1], s[4]) for s in b["opt_slots"]] == [(1, 0, 0)] and b["slot_pos"] == []


def _v3_examples():
    return [ex_copy(), ex_opt(), ex_list(), ex_opt("metric"), ex_list(("Google", "Meta", "Nvidia"))] * 3


@pytest.mark.parametrize("cached", [False, True])
def test_v3_loss_learns_on_tiny_model(tok, cached):
    model = tiny_v3(tok)
    ds = FeatureDataset(tok, _v3_examples(), cached=cached, markers=True, v3=True)
    items = [ds[i] for i in range(len(ds))]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng, first = random.Random(0), None
    for _ in range(60):
        batch = collate(items, tok, rng, cached=cached)
        loss, parts = model.loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert {"gate", "span", "opt", "count"} <= set(parts)
    assert loss.item() < 0.5 * first, (first, loss.item(), parts)


@pytest.mark.parametrize("cached", [False, True])
def test_predict_v3_end_to_end_well_formed(tok, cached):
    model = tiny_v3(tok)
    for ex in [ex_copy(), ex_opt(), ex_list()]:
        p = predict(model, tok, ex, T=2, cached=cached, markers=True, v3=True)
        names = {pp.name for t in ex.tools for pp in t.params}
        for c in p.calls:
            assert set(c.args) <= names
            assert all(0.0 <= v <= 1.0 for v in c.conf_args.values())


def test_overfit_v3_then_predict_copies_exact_text(tok):
    model = tiny_v3(tok)
    exs = [ex_copy()] * 8
    ds = FeatureDataset(tok, exs, markers=True, v3=True)
    items = [ds[i] for i in range(len(ds))]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(0)
    for _ in range(120):
        loss, _ = model.loss(collate(items, tok, rng))
        opt.zero_grad(); loss.backward(); opt.step()
    p = predict(model, tok, ex_copy(), T=2, markers=True, v3=True)
    assert p.calls and p.calls[0].name == "find_food" and p.calls[0].args.get("city") == "Boston"


def test_gate_trained_only_on_fully_masked_values_and_anchors_can_be_revealed(tok):
    # final-review finding: anchors were always masked (never revealed as at inference) and the copy gate trained
    # on values whose other slots were revealed (at inference it only ever sees a fully masked value)
    from actionman.data import _mask_slots
    from actionman.v3feat import v3_extras
    ex = ex_list(("Google", "Meta", "Nvidia", "Apple", "Tesla", "Amazon", "Intel"))
    a = build_pass_a_v3(tok, ex)
    b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True)
    revealed_anchor, clean_seen = False, False
    for seed in range(40):
        ids, targets = _mask_slots(b, random.Random(seed))
        clean = set(b["_clean"])
        masked = {p for p, _ in targets}
        for pos in b["slot_pos"]:
            if pos[0] in clean:
                clean_seen = True
                assert set(pos) <= masked                       # clean value: every slot masked
            elif pos[0] not in masked:
                revealed_anchor = True
        x = v3_extras([a], [(0, b)])
        assert sorted(x["anchor_p"].tolist()) == sorted(clean)  # gate/span loss only on clean anchors
    assert clean_seen and revealed_anchor


def test_legacy_mask_reproduces_original_v3(tok):
    # --v3-mask legacy: every anchor always masked and the gate trains on every anchor (the s10 v3 run)
    from actionman.data import _mask_slots
    from actionman.v3feat import v3_extras
    ex = ex_list(("Google", "Meta", "Nvidia"))
    a = build_pass_a_v3(tok, ex)
    b = build_pass_b_v3(tok, ex, a, gold_plan_v3(tok, ex, a), with_targets=True)
    _mask_slots(b, random.Random(0))                    # a clean-mode call leaves _clean behind
    for seed in range(20):
        ids, targets = _mask_slots(b, random.Random(seed), v3_mask="legacy")
        assert "_clean" not in b
        assert set(b["anchors"]) <= {p for p, _ in targets}
        x = v3_extras([a], [(0, b)])
        assert sorted(x["anchor_p"].tolist()) == sorted(b["anchors"])


def test_snap_word_widens_cut_words_only():
    from actionman.span import snap_word
    q = "convert 100km into US Dollars using miles"
    cut = lambda s: (q.index(s), q.index(s) + len(s))
    assert q[slice(*snap_word(q, *cut("US Doll")))] == "US Dollars"
    assert q[slice(*snap_word(q, *cut("ile")))] == "miles"
    assert q[slice(*snap_word(q, *cut("100")))] == "100"          # digit/letter edge is not widened
    assert q[slice(*snap_word(q, *cut("into US")))] == "into US"  # whole words unchanged
    assert q[slice(*snap_word(q, *cut("0")))] == "100"


def test_predict_v3_with_snap_runs(tok):
    m = tiny_v3(tok)
    for cached in (False, True):
        p = predict(m, tok, ex_copy(), T=2, cached=cached, markers=True, v3=True, snap=True)
        assert p.calls is not None


def test_require_never_abstains(tok):
    m = tiny_v3(tok)
    orig = m.decision_logits

    def no_call_wins(*a, **k):                    # make "no call" the argmax for every call slot
        out = orig(*a, **k).clone()
        out[..., -1] += 100.0
        return out
    m.decision_logits = no_call_wins
    free = predict(m, tok, ex_copy(), T=1, cached=True, markers=True, v3=True)
    forced = predict(m, tok, ex_copy(), T=1, cached=True, markers=True, v3=True, require=True)
    assert free.calls == []
    assert len(forced.calls) == 1 and forced.calls[0].name in {t.name for t in ex_copy().tools}
