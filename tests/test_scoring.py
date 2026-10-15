from actionman.schema import Call
from actionman.scoring import match_calls, names_match

GOLD = [{"get_weather": {"city": ["New York"], "unit": ["celsius", ""]}},
        {"area": {"radius": [5.0]}}]


def test_order_insensitive_and_string_norm():
    pred = [Call("area", {"radius": 5}), Call("get_weather", {"city": "new york"})]
    assert match_calls(pred, GOLD)


def test_int_float_coercion_and_optional():
    assert match_calls([Call("area", {"radius": "5"})], [{"area": {"radius": [5.0]}}])
    assert match_calls([Call("get_weather", {"city": "New York", "unit": "celsius"}), Call("area", {"radius": 5.0})], GOLD)


def test_extra_unknown_arg_fails():
    assert not match_calls([Call("area", {"radius": 5, "colour": "red"})], [{"area": {"radius": [5]}}])


def test_missing_required_fails():
    assert not match_calls([Call("area", {})], [{"area": {"radius": [5]}}])


def test_irrelevance():
    assert match_calls([], [])
    assert not match_calls([Call("area", {"radius": 1})], [])


def test_lists_and_names():
    assert match_calls([Call("f", {"xs": [3, 5]})], [{"f": {"xs": [[3, 5]]}}])
    assert not match_calls([Call("f", {"xs": [5, 3]})], [{"f": {"xs": [[3, 5]]}}])
    assert names_match([Call("b", {}), Call("a", {})], [{"a": {}}, {"b": {}}])
