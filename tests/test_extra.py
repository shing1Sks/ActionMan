import json

from actionman.extra import from_hermes, from_toolace, parse_python_calls


def test_python_calls_with_spaced_names():
    calls = parse_python_calls('[Market Trends API(trend_type="MARKET_INDEXES", country="us"), f(x=[1, 2])]',
                               ["Market Trends API", "f"])
    assert [(c.name, c.args) for c in calls] == [("Market Trends API", {"trend_type": "MARKET_INDEXES", "country": "us"}),
                                                 ("f", {"x": [1, 2]})]
    assert parse_python_calls("Sure, which city?", ["f"]) is None


def test_toolace_text_answer_is_no_call():
    tools = [{"name": "f", "description": "d", "parameters": {"type": "dict", "properties": {"x": {"type": "string"}}, "required": ["x"]}}]
    row = {"system": "Here is a list of functions in JSON format that you can invoke:\n" + json.dumps(tools) + "\nShould you decide",
           "conversations": [{"from": "user", "value": "hi"}, {"from": "assistant", "value": "Which x do you mean?"}]}
    ex = from_toolace(row, 0)
    assert ex.calls == [] and ex.tools[0].params[0].optional is False


def test_hermes_row():
    tools = [{"type": "function", "function": {"name": "cam", "description": "d", "parameters": {
        "type": "object", "properties": {"id": {"type": "string"}, "q": {"type": "string", "enum": ["720p", "4k"]}}, "required": ["id"]}}}]
    row = {"id": "x", "conversations": [
        {"from": "system", "value": "intro <tools>\n" + json.dumps(tools) + "\n</tools>"},
        {"from": "human", "value": "show front door in 4k"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "cam", "arguments": {"id": "front", "q": "4k"}}\n</tool_call>'}]}
    ex = from_hermes(row)
    assert ex.calls[0].args == {"id": "front", "q": "4k"}
    assert "one of 720p|4k" in ex.tools[0].params[1].description


def test_v4_converters():
    import json
    from actionman.extra import from_glaive, from_synth_apigen, from_toucan
    fn = {"name": "get_news", "description": "news", "parameters": {"type": "object", "properties": {
        "country": {"type": "string", "description": "c"}}, "required": ["country"]}}
    s = from_synth_apigen({"tools": json.dumps([{"type": "function", "function": fn}]), "query": "news for France",
                           "answers": json.dumps([{"name": "get_news", "arguments": {"country": "France"}}])}, 0)
    assert s.calls[0].args == {"country": "France"} and s.source == "synth"
    assert from_synth_apigen({"tools": json.dumps([fn]), "query": "q",
                              "answers": "The query cannot be answered with the provided tools."}, 1).calls == []
    assert from_synth_apigen({"tools": json.dumps([fn]), "query": "q",
                              "answers": "The given question lacks the parameters required by the function."}, 2) is None
    system = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n" + json.dumps(fn, indent=4)
    g = from_glaive({"system": system, "chat": "USER: Latest news in France?\n\n\nASSISTANT: <functioncall> "
                     "{\"name\": \"get_news\", \"arguments\": '{\"country\": \"France\"}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {}"}, 0)
    assert g.query == "Latest news in France?" and g.calls[0].args == {"country": "France"}
    assert from_glaive({"system": system, "chat": "USER: Order a pizza\n\n\nASSISTANT: I'm sorry, but I can't order pizza. <|endoftext|>"}, 1).calls == []
    assert from_glaive({"system": system, "chat": "USER: news\n\n\nASSISTANT: Sure, which country? <|endoftext|>"}, 2) is None
    fn2 = dict(fn, name="srv-get_news")
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"},
            {"role": "tool_call", "content": str({"name": "srv-get_news", "arguments": json.dumps({"country": "FR"})})},
            {"role": "tool_response", "content": "{}"}]
    t = from_toucan({"tools": json.dumps([{"type": "function", "function": fn2}]), "messages": json.dumps(msgs),
                     "question": "news for FR", "target_tools": "get_news"}, 0)
    assert t.calls[0].name == "srv-get_news" and t.calls[0].args == {"country": "FR"}
    assert from_toucan({"tools": json.dumps([{"type": "function", "function": fn2}]), "messages": json.dumps(msgs),
                        "question": "news", "target_tools": "get_news, get_news"}, 1) is None   # later step needed
