"""ActionMan as an arena driver, out of the box: the turn's observation is the query, the level's tools are the tools."""
from __future__ import annotations

import os

import torch

from actionman.evaluate import is_cached, load_model, uses_markers, uses_v3
from actionman.infer import predict
from actionman.schema import Example, Param, Tool

from .agents import Agent

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CKPT = os.path.join(ROOT, "runs", "hf", "runs", "s16-v3-5epb", "m")
JSON_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "list", "object": "dict"}


def to_tool(t: dict) -> Tool:
    ps = t["parameters"]
    req = set(ps.get("required", []))
    return Tool(t["name"], t.get("description", ""),
                [Param(n, JSON_TYPES.get(s.get("type"), "any"), n not in req, None, s.get("description", ""), s.get("enum"))
                 for n, s in ps["properties"].items()])


class ActionManAgent(Agent):
    name = "actionman"

    def __init__(self, ckpt: str = DEFAULT_CKPT, T: int = 3, require: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label = f"ActionMan ({'GPU' if self.device == 'cuda' else 'laptop CPU'})"
        self.note = "395M encoder + diffusion slot filler, out of the box" + ("" if self.device == "cuda" else "; CPU is slower than its GPU speed")
        torch.set_grad_enabled(False)
        self.model, self.tok = load_model(ckpt, self.device)
        self.flags = dict(cached=is_cached(ckpt), markers=uses_markers(ckpt), v3=uses_v3(ckpt))
        self.T, self.require = T, require     # require = tool_choice="required", as the API models get
        self.act("Dig down.",   # warm-up
                 [{"name": "dig", "description": "", "parameters": {"properties": {"direction": {"type": "string"}}}}], None)

    def act(self, state, tools, game):
        pred = predict(self.model, self.tok, Example(state, [to_tool(t) for t in tools], []), T=self.T, device=self.device,
                       require=self.require, **self.flags)
        return [{"name": c.name, "args": c.args} for c in pred.calls]
