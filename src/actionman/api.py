"""Public API: load a released ActionMan and turn (query, tools) into tool calls.

    from actionman.api import ActionMan
    am = ActionMan.from_pretrained("shing-dev/ActionMan-0.1")          # or a local export directory
    am.predict("Weather in Paris in celsius?", tools)                   # tools: OpenAI-style function schemas
    # -> [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}, "confidence": 0.97}]
"""
from __future__ import annotations

import json
import os

import torch

from . import cached as C
from .features import load_tokenizer
from .infer import predict as _predict
from .model import ActionMan as _Model
from .schema import Example, from_openai_tool


class ActionMan:
    def __init__(self, model, tok, flags: dict, device: str):
        self.model, self.tok, self.flags, self.device = model, tok, flags, device

    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None, compile: bool | None = None, token: str | None = None):
        """path: a local export dir or a Hugging Face repo id. compile (default: on GPU): padded static encoders
        under torch.compile, about 4x faster per call after a one-off warm-up."""
        from safetensors.torch import load_file
        from transformers import ModernBertConfig
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, token=token)
        cfg = json.load(open(os.path.join(path, "config.json")))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = load_tokenizer(path)
        model = _Model.from_encoder(ModernBertConfig.from_dict(cfg["encoder"]), tok, v3=cfg["flags"]["v3"])
        model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(path, "model.safetensors")).items()})
        model = model.to(device).eval()
        if compile if compile is not None else device == "cuda":     # process-wide swap, as in evaluate --compile
            C.encode_context, C.encode_slots = C.make_static(model.mlm.model)
        return cls(model, tok, cfg["flags"], device)

    @torch.no_grad()
    def predict(self, query: str, tools: list[dict], T: int = 3, require: bool = False) -> list[dict]:
        """Tool calls for `query`, best first. Empty list = no tool fits (unless require=True, like tool_choice="required").
        confidence = min(tool confidence, every argument's confidence). It ranks errors well enough to abstain on
        (docs/results/benchmark.md: the least confident 10% abstaining takes our eval from 78.6 to 85.1)."""
        ex = Example(query, [from_openai_tool(t) for t in tools], [])
        pred = _predict(self.model, self.tok, ex, T=T, device=self.device, require=require, **self.flags)
        return [{"name": c.name, "arguments": c.args,
                 "confidence": round(min([c.conf_tool, *c.conf_args.values()]), 4)} for c in pred.calls]
