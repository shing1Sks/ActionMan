"""Qwen3-0.6B fine-tuned on ActionMan's own training data (the same-data baseline) as a race driver.
Plain HF generate at batch 1 (not vLLM), so its latency is an upper bound; the benchmark plan measures a tuned server."""
from __future__ import annotations

import os

import torch

from actionman.schema import Example
from baseline.qwen import load, parse_calls, prompt_text

from .agents import Agent
from .am_agent import ROOT, to_tool

DEFAULT_CKPT = os.path.join(ROOT, "runs", "hf", "runs", "s5-qwen-v2", "qwen")


class QwenAgent(Agent):
    name = "qwen"

    def __init__(self, ckpt: str = DEFAULT_CKPT, max_new: int = 96):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label = f"Qwen3-0.6B SFT ({'GPU' if self.device == 'cuda' else 'laptop CPU'})"
        self.note = "same training data as ActionMan; HF generate, batch 1, call required"
        torch.set_grad_enabled(False)
        self.model, self.tok = load(ckpt, device=self.device)
        self.model.eval()
        self.max_new = max_new
        self.act("Dig down.", [{"name": "dig", "description": "", "parameters": {"properties": {"direction": {"type": "string"}}}}], None)

    def act(self, state, tools, game):
        # the answer is prefilled with "<tool_call>": a call is required, as with tool_choice="required" (GPT) and require=True (ActionMan)
        text = prompt_text(self.tok, Example(state, [to_tool(t) for t in tools], [])) + "<tool_call>\n"
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        g = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
        out = "<tool_call>\n" + self.tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=False)
        return [{"name": c.name, "args": c.args} for c in parse_calls(out)]
