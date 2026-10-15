"""Unified tool/call schema, converters from xLAM and BFCL, compact signatures, value <-> text."""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any

TYPES = ("str", "int", "float", "bool", "list", "dict", "any")


@dataclass
class Param:
    name: str
    type: str = "any"
    optional: bool = False
    default: Any = None
    description: str = ""
    options: list | None = None          # declared enum values (v3 option slots)


@dataclass
class Tool:
    name: str
    description: str = ""
    params: list[Param] = field(default_factory=list)


@dataclass
class Call:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Example:
    query: str
    tools: list[Tool]
    calls: list[Call]
    id: str = ""
    source: str = ""
    # gold in BFCL "possible answer" form: [{func: {arg: [acceptable values]}}]; "" = may be omitted
    gold: list[dict] = field(default_factory=list)


# ---------- type normalisation ----------

def norm_type(t: Any) -> tuple[str, bool]:
    """Map xLAM ('str, optional', 'List[int]') or JSON-schema ('integer') types to (TYPE, optional)."""
    s = str(t or "any").strip()
    low = s.lower()
    optional = "optional" in low
    head = low.split(",")[0].split("[")[0].strip()
    table = {
        "str": "str", "string": "str",
        "int": "int", "integer": "int",
        "float": "float", "number": "float", "double": "float",
        "bool": "bool", "boolean": "bool",
        "list": "list", "array": "list", "tuple": "list", "set": "list",
        "dict": "dict", "object": "dict",
    }
    return table.get(head, "any"), optional


# ---------- converters ----------

def from_xlam(row: dict) -> Example:
    tools_raw = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
    tools = []
    for t in tools_raw:
        params = []
        for pname, spec in (t.get("parameters") or {}).items():
            spec = spec or {}
            ty, opt = norm_type(spec.get("type"))
            default = spec.get("default")
            if default == "":
                default = None
            params.append(Param(pname, ty, opt, default, spec.get("description", "") or ""))
        tools.append(Tool(t["name"], t.get("description", "") or "", params))
    calls = [Call(a["name"], dict(a.get("arguments") or {})) for a in answers]
    ex = Example(row["query"], tools, calls, id=f"xlam_{row.get('id', '')}", source="xlam")
    ex.gold = gold_from_calls(tools, calls)
    return ex


def from_bfcl(q_row: dict, answer_row: dict | None, category: str) -> Example:
    query = q_row["question"][0][-1]["content"] if q_row["question"] and q_row["question"][0] else ""
    tools = []
    for f in q_row["function"]:
        props = (f.get("parameters") or {}).get("properties", {}) or {}
        required = set((f.get("parameters") or {}).get("required", []) or [])
        params = []
        for pname, spec in props.items():
            ty, _ = norm_type(spec.get("type"))
            params.append(Param(pname, ty, pname not in required, spec.get("default"), spec.get("description", "") or "",
                                spec.get("enum") or None))
        tools.append(Tool(f["name"], f.get("description", "") or "", params))
    gold = answer_row["ground_truth"] if answer_row else []
    calls = []
    for g in gold:
        (name, args), = g.items()
        calls.append(Call(name, {k: v[0] for k, v in args.items() if v and v[0] != ""}))
    return Example(query, tools, calls, id=q_row["id"], source=f"bfcl_{category}", gold=gold)


def gold_from_calls(tools: list[Tool], calls: list[Call]) -> list[dict]:
    """Build BFCL-style acceptable-value gold from exact xLAM answers."""
    by_name = {t.name: t for t in tools}
    gold = []
    for c in calls:
        accept: dict[str, list] = {k: [v] for k, v in c.args.items()}
        tool = by_name.get(c.name)
        if tool:
            for p in tool.params:
                if p.name in accept:
                    if p.optional and p.default is not None and _same(accept[p.name][0], p.default):
                        accept[p.name].append("")
                elif p.optional or p.default is not None:
                    accept[p.name] = [""] + ([p.default] if p.default is not None else [])
        gold.append({c.name: accept})
    return gold


def _same(a: Any, b: Any) -> bool:
    try:
        return a == b or float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


MAX_OPTIONS = 16


def param_options(p: Param) -> list | None:
    """Values a v3 option slot picks from: bools -> [True, False], declared enums of <= MAX_OPTIONS, else None."""
    if p.type == "bool":
        return [True, False]
    if p.options and len(p.options) <= MAX_OPTIONS:
        return list(p.options)
    return None


# ---------- compact signature ----------

def _trim(text: str, words: int) -> str:
    w = " ".join(str(text).split()).split(" ")
    return " ".join(w[:words]) if words > 0 else ""


def signature_pieces(tool: Tool, desc_words: int = 8) -> list[tuple[str, str, int]]:
    """Pieces of `name(p:type=default "desc", q?:int) | description`.
    Each piece is (text, kind, param_index); kind in {'name', 'param', 'text'}.
    The 'param' piece is exactly the parameter name so its first token can be located."""
    pieces: list[tuple[str, str, int]] = [(tool.name, "name", -1), ("(", "text", -1)]
    for k, p in enumerate(tool.params):
        if k:
            pieces.append((", ", "text", -1))
        pieces.append((p.name, "param", k))
        spec = ("?" if p.optional else "") + ":" + p.type
        if p.default is not None:
            spec += "=" + value_to_text(p.default)
        d = _trim(p.description, desc_words)
        if d:
            spec += f' "{d}"'
        pieces.append((spec, "text", -1))
    pieces.append((")", "text", -1))
    d = _trim(tool.description, max(desc_words + 4, 0) if desc_words > 0 else 0)
    if d:
        pieces.append((" | " + d, "text", -1))
    return pieces


def signature(tool: Tool, desc_words: int = 8) -> str:
    return "".join(t for t, _, _ in signature_pieces(tool, desc_words))


# ---------- values ----------

def value_to_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    return json.dumps(v, ensure_ascii=False)


def text_to_value(s: str, ty: str) -> Any:
    s = s.strip()
    if ty == "str":
        return s
    if ty == "bool":
        return s.lower() in ("true", "1", "yes")
    if ty in ("int", "float"):
        try:
            f = float(s)
            if ty == "int" and f.is_integer():
                return int(f)
            return f
        except ValueError:
            pass
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(s)
        except Exception:
            continue
    return s


# ---------- OpenAI-style tool (for the LLM baseline) ----------

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
               "list": "array", "dict": "object", "any": "string"}


def from_openai_tool(t: dict) -> Tool:
    """OpenAI-style tool ({"type": "function", "function": {...}} or the bare function dict) -> Tool."""
    f = t.get("function", t)
    ps = f.get("parameters") or {}
    req = set(ps.get("required", []))
    params = []
    for name, s in (ps.get("properties") or {}).items():
        ty, _ = norm_type(s.get("type"))
        params.append(Param(name, ty, name not in req, s.get("default"), s.get("description", "") or "", s.get("enum")))
    return Tool(f["name"], f.get("description", "") or "", params)


def to_openai_tool(tool: Tool) -> dict:
    props = {}
    for p in tool.params:
        spec = {"type": _JSON_TYPES[p.type], "description": p.description}
        if p.default is not None:
            spec["default"] = p.default
        props[p.name] = spec
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description,
        "parameters": {"type": "object", "properties": props,
                       "required": [p.name for p in tool.params if not p.optional]}}}


# ---------- (de)serialisation for jsonl ----------

def example_to_dict(ex: Example) -> dict:
    return {"id": ex.id, "source": ex.source, "query": ex.query, "gold": ex.gold,
            "tools": [{"name": t.name, "description": t.description,
                       "params": [vars(p) for p in t.params]} for t in ex.tools],
            "calls": [vars(c) for c in ex.calls]}


def example_from_dict(d: dict) -> Example:
    tools = [Tool(t["name"], t["description"], [Param(**p) for p in t["params"]]) for t in d["tools"]]
    return Example(d["query"], tools, [Call(**c) for c in d["calls"]], d["id"], d["source"], d.get("gold", []))
