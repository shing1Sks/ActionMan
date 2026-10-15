import torch

from actionman.api import ActionMan as API
from actionman.export_hf import export
from actionman.infer import predict
from actionman.schema import from_openai_tool, to_openai_tool
from tests.test_v3 import ex_copy, tiny_v3


def test_openai_tool_round_trip():
    t = ex_copy().tools[0]
    back = from_openai_tool(to_openai_tool(t))
    assert back.name == t.name and [(p.name, p.type, p.optional) for p in back.params] == [(p.name, p.type, p.optional) for p in t.params]
    bare = from_openai_tool({"name": "f", "parameters": {"properties": {"u": {"type": "string", "enum": ["c", "f"]}}, "required": ["u"]}})
    assert bare.params[0].options == ["c", "f"] and not bare.params[0].optional


def test_export_then_from_pretrained_predicts_the_same(tok, tmp_path):
    m = tiny_v3(tok).eval()
    flags = dict(cached=True, markers=True, v3=True)
    export(m, tok, flags, str(tmp_path))
    am = API.from_pretrained(str(tmp_path), device="cpu", compile=False)
    ex = ex_copy()
    want = predict(m.to(torch.bfloat16).float(), tok, ex, T=2, device="cpu", **flags)   # export stores bf16 weights
    got = am.predict(ex.query, [to_openai_tool(t) for t in ex.tools], T=2)
    assert [(c["name"], c["arguments"]) for c in got] == [(c.name, c.args) for c in want.calls]
    assert all(0 <= c["confidence"] <= 1 for c in got)
    assert (tmp_path / "code" / "actionman" / "api.py").exists()


def test_bundled_code_is_self_contained():
    import ast
    import os
    from actionman.export_hf import CODE
    here = os.path.dirname(os.path.abspath(__import__("actionman").__file__))
    for f in CODE:
        for n in ast.walk(ast.parse(open(os.path.join(here, f), encoding="utf-8").read())):
            if isinstance(n, ast.ImportFrom) and n.level == 1:
                mods = [n.module.split(".")[0]] if n.module else [a.name for a in n.names]
                assert all(m + ".py" in CODE for m in mods), (f, mods)
