"""Converters for extra training data (Stage 5): Hermes function-calling (single turn) and ToolACE (first turn)."""
from __future__ import annotations

import ast
import json
import re

from .schema import Call, Example, Param, Tool, gold_from_calls, norm_type


def _tool_from_schema(f: dict) -> Tool:
    params_spec = f.get("parameters") or {}
    props = params_spec.get("properties", {}) or {}
    required = set(params_spec.get("required") or [])
    params = []
    for name, spec in props.items():
        spec = spec or {}
        ty, _ = norm_type(spec.get("type"))
        desc = spec.get("description", "") or ""
        if spec.get("enum"):
            desc = (desc + " one of " + "|".join(map(str, spec["enum"][:8]))).strip()
        params.append(Param(name, ty, name not in required, spec.get("default"), desc, spec.get("enum") or None))
    return Tool(f["name"], f.get("description", "") or "", params)


def from_hermes(row: dict) -> Example | None:
    conv = row["conversations"]
    sys_msg = next((c["value"] for c in conv if c["from"] == "system"), "")
    blocks = re.findall(r"<tools>\s*(.*?)\s*</tools>", sys_msg, re.S)   # prompt also mentions '<tools> </tools>'
    m = max(blocks, key=len) if blocks else None
    human = next((c["value"] for c in conv if c["from"] == "human"), None)
    gpt = next((c["value"] for c in conv if c["from"] == "gpt"), None)
    if not (m and human and gpt):
        return None
    try:
        tools = [_tool_from_schema(t.get("function", t)) for t in json.loads(m)]
        calls = [Call(d["name"], d.get("arguments") or {}) for d in
                 (json.loads(x) for x in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", gpt, re.S))]
    except Exception:
        return None
    ex = Example(human, tools, calls, id=f"hermes_{row.get('id', '')}", source="hermes")
    ex.gold = gold_from_calls(tools, calls)
    return ex


def parse_python_calls(text: str, names: list[str]) -> list[Call] | None:
    """Parse '[Func A(x=1, y="b"), g(z=[1,2])]' where function names may contain spaces/dots."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    alias = {}
    for i, n in enumerate(sorted(names, key=len, reverse=True)):
        key = f"__f{i}"
        if n + "(" in text:
            text = text.replace(n + "(", key + "(")
            alias[key] = n
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.List):
        return None
    calls = []
    for node in tree.body.elts:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            return None
        name = alias.get(node.func.id, node.func.id)
        try:
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        except Exception:
            return None
        calls.append(Call(name, args))
    return calls


def from_toolace(row: dict, idx: int) -> Example | None:
    sys_msg = row.get("system", "")
    start = sys_msg.find("[{")
    if start < 0:
        return None
    try:
        tools_raw, _ = json.JSONDecoder().raw_decode(sys_msg[start:])
        tools = [_tool_from_schema(t) for t in tools_raw]
    except Exception:
        return None
    conv = row["conversations"]
    if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
        return None
    calls = parse_python_calls(conv[1]["value"], [t.name for t in tools])
    if calls is None:
        if conv[1]["value"].strip().startswith("["):
            return None          # malformed call text: skip rather than teach "no call"
        calls = []               # assistant answered in text (clarify / can't help) -> no call
    ex = Example(conv[0]["value"], tools, calls, id=f"toolace_{idx}", source="toolace")
    ex.gold = gold_from_calls(tools, calls)
    return ex


# ---------- v4 extra data (2026-10-15): Synth-APIGen, Glaive v2, Toucan ----------
# (nvidia/When2Call was checked and left out: 5,162 of its 5,191 queries are xLAM training queries paired with
# unusable tools, the same thing as our existing xlam_neg / xlam_hardneg rows.)

_REFUSAL = re.compile(r"\b(sorry|unable|can't|cannot|can not|don't have|not able)\b", re.I)


def _fn(t: dict) -> dict:
    return t.get("function", t)


def _example(query: str, tools: list[Tool], calls: list[Call], idx, source: str) -> Example | None:
    names = {t.name for t in tools}
    if not query.strip() or not tools or any(c.name not in names for c in calls):
        return None
    ex = Example(query.strip(), tools, calls, id=f"{source}_{idx}", source=source)
    ex.gold = gold_from_calls(tools, calls)
    return ex


def from_synth_apigen(row: dict, idx) -> Example | None:
    """argilla/Synth-APIGen-v0.1: xLAM-style JSON answers; 'cannot be answered with the provided tools' -> no call.
    'lacks the parameters' rows are skipped (asking for details is not a no-call we want to teach)."""
    try:
        tools = [_tool_from_schema(_fn(t)) for t in json.loads(row["tools"])]
    except Exception:
        return None
    try:
        ans = json.loads(row["answers"])
    except Exception:
        ans = None
    if isinstance(ans, list):
        calls = [Call(a["name"], dict(a.get("arguments") or {})) for a in ans if isinstance(a, dict) and "name" in a]
        if len(calls) != len(ans) or not calls:
            return None
    elif "cannot be answered" in str(row["answers"]):
        calls = []
    else:
        return None
    return _example(row["query"], tools, calls, idx, "synth")


def _glaive_tools(system: str) -> list[Tool]:
    out, dec, i = [], json.JSONDecoder(), system.find("{")
    while 0 <= i < len(system):
        f, j = dec.raw_decode(system[i:])
        out.append(_tool_from_schema(f))
        i = system.find("{", i + j)
    return out


def from_glaive(row: dict, idx) -> Example | None:
    """glaiveai/glaive-function-calling-v2, first turn: a <functioncall> -> one call; a refusal -> no call;
    anything else (clarifying question, direct answer) is skipped."""
    try:
        tools = _glaive_tools(row["system"])
    except Exception:
        return None
    m = re.match(r"\s*USER:(.*?)\n\n\nASSISTANT:(.*?)(?:\n\n\n|$)", row["chat"], re.S)
    if not (tools and m):
        return None
    user, reply = m.group(1).strip(), m.group(2).replace("<|endoftext|>", "").strip()
    if "<functioncall>" in reply:
        c = re.search(r'"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*)\'\s*\}', reply, re.S)
        if not c:
            return None
        try:
            args = json.loads(c.group(2))
        except Exception:
            return None
        if not isinstance(args, dict):
            return None
        calls = [Call(c.group(1), args)]
    elif _REFUSAL.search(reply) and "?" not in reply.split(".")[0]:
        calls = []
    else:
        return None
    return _example(user, tools, calls, idx, "glaive")


def from_toucan(row: dict, idx) -> Example | None:
    """Agent-Ark/Toucan-1.5M (MCP tools): the calls of the first step, kept only when that step already makes every
    target call (later steps depend on tool results, which a one-shot model cannot see)."""
    try:
        tools = [_tool_from_schema(_fn(t)) for t in json.loads(row["tools"])]
        msgs = json.loads(row["messages"])
    except Exception:
        return None
    first = []
    for m in msgs[1:]:
        if m["role"] == "tool_call":
            first.append(m)
        elif m["role"] == "tool_response" or first:
            break
    calls = []
    try:
        for m in first:
            d = ast.literal_eval(m["content"]) if isinstance(m["content"], str) else m["content"]
            args = json.loads(d["arguments"]) if isinstance(d["arguments"], str) else d["arguments"]
            if not isinstance(args, dict):
                return None
            calls.append(Call(d["name"], args))
    except Exception:
        return None
    target = [t.strip() for t in str(row.get("target_tools") or "").split(",") if t.strip()]
    if not calls or len(calls) != len(target) or not all(any(c.name.endswith(t) for c in calls) for t in target):
        return None
    return _example(row["question"], tools, calls, idx, "toucan")
