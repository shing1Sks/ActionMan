from actionman.features import (MAX_CALLS, OVERFLOW, build_pass_a, build_pass_b, bucket_of, gold_plan)
from actionman.schema import Call

from conftest import make_example


def test_pass_a_layout(tok, example):
    a = build_pass_a(tok, example)
    ids = a["input_ids"]
    assert sum(1 for x in ids if x == tok.am["[CALL]"]) == MAX_CALLS
    assert all(ids[p] == tok.am["[T]"] for p in a["t_pos"]) and len(a["t_pos"]) == 2
    assert all(ids[p] == tok.am["[CALL]"] for p in a["call_pos"])
    assert a["dec_labels"] == [0, 2, 2, 2]          # web_search, then NONE (= n_tools)
    q_tok = tok.convert_ids_to_tokens(ids[a["arg_pos"][(0, 0)]])
    assert q_tok.lstrip("Ġ") == "q"


def test_buckets():
    assert [bucket_of(n) for n in (0, 3, 4, 5, 9, 32, 33)] == [1, 1, 1, 2, 3, 4, OVERFLOW]


def test_pass_b_slots_and_pad(tok, example):
    a = build_pass_a(tok, example)
    plan = gold_plan(a, example)
    b = build_pass_b(tok, example, a, plan, with_targets=True)
    (q_slots, k_slots) = b["slot_pos"]
    assert len(k_slots) == 4                          # "3" -> bucket 1 -> 4 slots
    assert b["slot_targets"][1][1:] == [tok.pad_token_id] * 3
    assert all(b["input_ids"][p] == tok.mask_token_id for p in q_slots + k_slots)
    assert tok.decode(b["slot_targets"][0]).replace("[PAD]", "").strip() == "best ramen near shibuya"


def test_overflow_and_absent(tok):
    ex = make_example(q="word " * 40)
    ex.calls[0].args.pop("k")
    a = build_pass_a(tok, ex)
    assert [b for (_, _, _, b) in a["len_items"]] == [OVERFLOW, 0]
    b = build_pass_b(tok, ex, a, gold_plan(a, ex), with_targets=True)
    assert b["slot_pos"] == []


def test_more_than_four_calls_not_trained(tok):
    ex = make_example(extra_calls=[Call("get_weather", {"city": c}) for c in "abcd"])
    a = build_pass_a(tok, ex)
    assert a["dec_labels"] == [-100] * 4


def test_zero_param_call_keeps_alignment(tok):
    from actionman.schema import Example, Param, Tool
    tools = [Tool("ping", "no params", []), Tool("get_client", "client", [Param("client", "str")])]
    ex = Example("ping then get acme", tools, [Call("ping", {}), Call("get_client", {"client": "acme"})])
    a = build_pass_a(tok, ex)
    plan = gold_plan(a, ex)
    assert plan == [(0, {}), (1, {0: 1})]
    b = build_pass_b(tok, ex, a, plan, with_targets=True)
    assert tok.decode(b["slot_targets"][0]).replace("[PAD]", "").strip() == "acme"


def test_markers_distinguish_repeated_calls(tok):
    ex = make_example(extra_calls=[Call("web_search", {"q": "pizza", "k": 2})])
    a = build_pass_a(tok, ex)
    b = build_pass_b(tok, ex, a, gold_plan(a, ex), with_targets=True, markers=True)
    text = tok.decode(b["input_ids"])
    assert "#1 web_search(" in text and "#2 web_search(" in text
    plain = build_pass_b(tok, ex, a, gold_plan(a, ex), with_targets=True)
    assert "#1" not in tok.decode(plain["input_ids"])
