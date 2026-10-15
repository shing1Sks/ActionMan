import json

from bench.bfcl_actionman import SKIPPED, category, entry_to_example, example_to_bfcl, fit, flatten, to_result
from actionman.features import MAX_LEN_A, build_pass_a
from actionman.schema import Call, Example, Param, Tool

FN = [{"name": "get_user_info", "description": "Retrieve details.",
       "parameters": {"type": "dict", "required": ["user_id"],
                      "properties": {"user_id": {"type": "integer", "description": "id"},
                                     "special": {"type": "string", "description": "", "default": "none"},
                                     "extra": {"type": "any", "description": ""}}}}]


def entry(msgs, eid="live_simple_0-0-0"):
    return {"id": eid, "question": [msgs], "function": FN}


def test_category_from_ids():
    assert category("live_simple_0-0-0") == "live_simple"
    assert category("simple_python_12") == "simple_python"
    assert category("irrelevance_3") == "irrelevance"


def test_flatten_lone_user_stays_bare_and_last_user_mode():
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yes?"}, {"role": "user", "content": "user 7890"}]
    assert flatten(msgs[1:2]) == "hi"
    assert flatten(msgs) == "System: Be brief.\nUser: hi\nAssistant: yes?\nUser: user 7890"
    assert flatten(msgs, "last") == "user 7890"


def test_entry_to_example_keeps_schema_and_category():
    ex = entry_to_example(entry([{"role": "user", "content": "details for 7890"}]))
    assert ex.query == "details for 7890" and ex.source == "live_simple"
    assert [(p.name, p.type, p.optional) for p in ex.tools[0].params] == [
        ("user_id", "int", False), ("special", "str", True), ("extra", "any", True)]


def test_to_result_types_strings_by_schema_and_java():
    res = to_result([Call("get_user_info", {"user_id": 7890, "extra": 5})], FN, "live_simple")
    assert res == [{"get_user_info": json.dumps({"user_id": 7890, "extra": "5"})}]
    java = to_result([Call("get_user_info", {"user_id": 7890})], FN, "simple_java")
    assert json.loads(java[0]["get_user_info"]) == {"user_id": "7890"}
    assert to_result([], FN, "irrelevance") == []
    sets = to_result([Call("get_user_info", {"user_id": 1, "tags": {"b", "a"}, "xy": (1, 2)})], FN, "live_simple")
    assert json.loads(sets[0]["get_user_info"]) == {"user_id": 1, "tags": ["a", "b"], "xy": [1, 2]}
    assert list(SKIPPED[0]) == ["__skipped__"]


def test_fit_truncates_query_from_the_front(tok):
    ex = entry_to_example(entry([{"role": "system", "content": "filler " * 1200},
                                 {"role": "user", "content": "details for user 7890"}]))
    assert build_pass_a(tok, ex) is None
    short, status = fit(tok, ex)
    assert status == "truncated" and short.query.endswith("User: details for user 7890")
    assert len(build_pass_a(tok, short)["input_ids"]) <= MAX_LEN_A
    assert fit(tok, entry_to_example(entry([{"role": "user", "content": "hi"}])))[1] == "ok"


def test_example_to_bfcl_xlam_gold_and_item_types():
    tools = [Tool("f", "d", [Param("ids", "list"), Param("opts", "dict", True), Param("n", "int", True, 5)])]
    ex = Example("q", tools, [], "xlam_1", "xlam", [{"f": {"ids": [[1, 2]], "opts": [{"a": 1}], "n": [5, ""]}}])
    e, gold, cat = example_to_bfcl(ex)
    props = e["function"][0]["parameters"]["properties"]
    assert cat == "multiple" and e["id"] == "own_xlam_1" and e["orig_id"] == "xlam_1"
    assert props["ids"] == {"type": "array", "description": "", "items": {"type": "integer"}}
    assert props["n"]["default"] == 5 and e["function"][0]["parameters"]["required"] == ["ids"]
    assert gold == [{"f": {"ids": [[1, 2]], "opts": [{"a": [1]}], "n": [5, ""]}}]
    assert example_to_bfcl(Example("q", tools, [], "neg_1", "xlam_neg", []))[2] == "irrelevance"


def test_example_to_bfcl_uses_v4_docs_for_bfcl_rows():
    ex = Example("q", [Tool("g")], [], "simple_3", "bfcl_simple", [{"g": {}}])
    e, gold, cat = example_to_bfcl(ex, {"simple_python_3": FN})
    assert cat == "simple" and e["function"] is FN and gold == [{"g": {}}]
