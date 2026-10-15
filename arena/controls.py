"""Control schemes shared by every game: renamed tool schemas per level, and translation of a model's call
back to a canonical action (kind, args). Level 0-1: canonical names. Level 2: tool and parameter names renamed.
Level 3: enum values renamed too, while the observation keeps the plain words."""
from __future__ import annotations

import random


class Scheme:
    def __init__(self, spec: dict, level: int = 0, seed: int = 0, game: str = ""):
        r = random.Random(f"controls-{game}-{level}-{seed}")
        self.spec = spec
        self.tool = {k: (r.choice(spec["tool_aliases"][k]) if level >= 2 else k) for k in spec["tools"]}
        self.param = {k: (r.choice(v) if level >= 2 else k) for k, v in spec["param_aliases"].items()}
        self.value = {g: (r.choice(v["sets"]) if level >= 3 else {c: c for c in v["canon"]}) for g, v in spec["values"].items()}
        self._tool_back = {v: k for k, v in self.tool.items()}

    def _pname(self, p):
        return self.param.get(p, p)

    def tools(self) -> list[dict]:
        """OpenAI-style function schemas in this level's names."""
        out = []
        for k, t in self.spec["tools"].items():
            props, req = {}, []
            for p, s in t["params"].items():
                d = {"type": s["type"], "description": s["desc"]}
                if "values" in s:
                    d["enum"] = [self.value[s["values"]][c] for c in self.spec["values"][s["values"]]["canon"]]
                props[self._pname(p)] = d
                if not s.get("optional"):
                    req.append(self._pname(p))
            out.append({"name": self.tool[k], "description": t["desc"],
                        "parameters": {"type": "object", "properties": props, "required": req}})
        return out

    def translate(self, call: dict | None):
        """Model call {name, args} -> ((kind, canonical args), "") or (None, error)."""
        if not call:
            return None, "no call"
        kind = self._tool_back.get(call.get("name"))
        if kind is None:
            return None, f"unknown tool {call.get('name')!r}"
        params, args, out = self.spec["tools"][kind]["params"], dict(call.get("args") or {}), {}
        back = {self._pname(p): p for p in params}
        extra = [a for a in args if a not in back]
        if extra:
            return None, f"unknown argument {extra[0]!r}"
        for p, s in params.items():
            name = self._pname(p)
            if name not in args:
                if not s.get("optional"):
                    return None, f"missing {name!r}"
                out[p] = s.get("default")
                continue
            v = args[name]
            if "values" in s:
                inv = {a.lower(): c for c, a in self.value[s["values"]].items()}
                c = inv.get(str(v).strip().lower())
                if c is None:
                    return None, f"bad {name} {v!r}"
                out[p] = c
            elif s["type"] == "integer":
                try:
                    out[p] = min(s.get("max", 10 ** 9), max(s.get("min", -10 ** 9), int(float(v))))
                except (TypeError, ValueError):
                    return None, f"bad {name} {v!r}"
            else:
                out[p] = v
        return (kind, out), ""

    def encode(self, action: tuple) -> dict:
        """Canonical (kind, args) -> a call in this level's names (used by the perfect driver)."""
        kind, args = action
        params = self.spec["tools"][kind]["params"]
        out = {}
        for p, v in (args or {}).items():
            if v is None:
                continue
            s = params[p]
            out[self._pname(p)] = self.value[s["values"]][v] if "values" in s else v
        return {"name": self.tool[kind], "args": out}

    def renames(self) -> dict:
        """What changed this level, for the viewer ribbon."""
        return {"tools": {k: v for k, v in self.tool.items() if k != v}, "params": {k: v for k, v in self.param.items() if k != v},
                "values": {g: {c: a for c, a in m.items() if c != a} for g, m in self.value.items()}}


def call_text(call: dict | None) -> str:
    if not call:
        return "(no call)"
    return f"{call.get('name')}(" + ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items()) + ")"
