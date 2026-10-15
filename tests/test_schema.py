import json

import pytest

from actionman.schema import (Call, Param, Tool, from_bfcl, from_xlam, norm_type, signature,
                              text_to_value, value_to_text, example_to_dict, example_from_dict)

XLAM_ROW = {
    "id": 0, "query": "Where can I find live giveaways for beta access and games?",
    "answers": json.dumps([{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
                           {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}]),
    "tools": json.dumps([{"name": "live_giveaways_by_type", "description": "Retrieve live giveaways.",
                          "parameters": {"type": {"description": "The type of giveaways.", "type": "str", "default": "game"}}}]),
}


def test_xlam_row_converts():
    ex = from_xlam(XLAM_ROW)
    assert len(ex.tools) == 1 and len(ex.calls) == 2
    assert ex.calls[1].args == {"type": "game"}
    assert ex.tools[0].params[0].default == "game"


def test_norm_type():
    assert norm_type("str, optional") == ("str", True)
    assert norm_type("List[int]") == ("list", False)
    assert norm_type("integer") == ("int", False)
    assert norm_type("Callable[[float], float]") == ("any", False)


def test_bfcl_required_drives_optional():
    q = {"id": "simple_0", "question": [[{"role": "user", "content": "area of triangle base 10 height 5"}]],
         "function": [{"name": "calc_area", "description": "Area.", "parameters": {"type": "dict",
                       "properties": {"base": {"type": "integer"}, "unit": {"type": "string"}}, "required": ["base"]}}]}
    a = {"id": "simple_0", "ground_truth": [{"calc_area": {"base": [10], "unit": ["units", ""]}}]}
    ex = from_bfcl(q, a, "simple")
    assert [p.optional for p in ex.tools[0].params] == [False, True]
    assert ex.calls[0].args == {"base": 10, "unit": "units"}


def test_signature_exact():
    t = Tool("web_search", "Search the web for pages",
             [Param("q", "str", False, None, "search query"), Param("k", "int", True, 5, "")])
    assert signature(t) == 'web_search(q:str "search query", k?:int=5) | Search the web for pages'
    assert signature(t, desc_words=0) == "web_search(q:str, k?:int=5)"


@pytest.mark.parametrize("v,ty", [("ramen", "str"), (3, "int"), (2.5, "float"), (True, "bool"),
                                  ([3, 5], "list"), ({"a": 1}, "dict")])
def test_value_round_trip(v, ty):
    assert text_to_value(value_to_text(v), ty) == v


def test_example_dict_round_trip():
    ex = from_xlam(XLAM_ROW)
    assert example_from_dict(example_to_dict(ex)) == ex


def test_param_options_bool_and_enum():
    from actionman.schema import Param, param_options
    assert param_options(Param("x", "bool")) == [True, False]
    assert param_options(Param("u", "str", options=["metric", "imperial"])) == ["metric", "imperial"]
    assert param_options(Param("n", "int")) is None
    assert param_options(Param("b", "str", options=[str(i) for i in range(40)])) is None   # too many -> generate


def test_bfcl_enum_survives_roundtrip():
    from actionman.schema import example_from_dict, example_to_dict, from_bfcl
    q = {"id": "s_0", "question": [[{"role": "user", "content": "convert"}]],
         "function": [{"name": "f", "description": "d", "parameters": {"type": "dict", "properties": {
             "unit": {"type": "string", "enum": ["metric", "imperial"]}}, "required": ["unit"]}}]}
    ex = from_bfcl(q, {"ground_truth": [{"f": {"unit": ["metric"]}}]}, "simple")
    assert example_from_dict(example_to_dict(ex)).tools[0].params[0].options == ["metric", "imperial"]
