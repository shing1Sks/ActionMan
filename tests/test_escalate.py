from bench.escalate import assisted_prefix, needs_escalation
from actionman.infer import PredCall, Prediction
from actionman.schema import Call, Example, Param, Tool


def ex_bmi():
    bmi = Tool("calculate_bmi", "bmi", [Param("weight", "int"), Param("height", "int"), Param("unit", "str", True)])
    other = Tool("get_weather", "weather", [Param("city", "str")])
    return Example("bmi for 85 kg and 5ft10", [bmi, other], [])


def pred_bmi(h_conf=0.62, tool_conf=1.0):
    return Prediction([PredCall("calculate_bmi", {"weight": 85, "height": 60, "unit": "metric"}, tool_conf,
                                {"weight": 1.0, "height": h_conf, "unit": 0.99})], 3)


def test_assisted_prefix_prefills_confident_args_and_stops_at_shaky_one():
    tools, prefix = assisted_prefix(pred_bmi(), ex_bmi(), theta=0.7)
    assert [t.name for t in tools] == ["calculate_bmi"]
    assert prefix.startswith("<tool_call>\n")
    assert prefix.endswith('"arguments": {"weight": 85, "unit": "metric", "height": ')


def test_no_escalation_when_all_confident():
    assert not needs_escalation(pred_bmi(0.95), 0.7)
    assert assisted_prefix(pred_bmi(0.95), ex_bmi(), theta=0.7) is None


def test_low_tool_confidence_gives_full_tool_list_and_no_prefix():
    tools, prefix = assisted_prefix(pred_bmi(0.95, tool_conf=0.4), ex_bmi(), theta=0.7)
    assert [t.name for t in tools] == ["calculate_bmi", "get_weather"] and prefix == ""


def test_confident_calls_written_first_then_partial():
    p = pred_bmi()
    p.calls.insert(0, PredCall("get_weather", {"city": "Pune"}, 0.99, {"city": 0.98}))
    tools, prefix = assisted_prefix(p, ex_bmi(), theta=0.7)
    assert prefix.startswith('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Pune"}}\n</tool_call>\n<tool_call>\n')
    assert [t.name for t in tools] == ["get_weather", "calculate_bmi"]


def test_parse_merged_output_round_trips():
    from bench.escalate import finish
    _, prefix = assisted_prefix(pred_bmi(), ex_bmi(), theta=0.7)
    calls = finish(prefix, '70}}\n</tool_call><|im_end|>')
    assert [(c.name, c.args) for c in calls] == [("calculate_bmi", {"weight": 85, "unit": "metric", "height": 70})]
