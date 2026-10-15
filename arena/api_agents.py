"""API players, run from the laptop so keys never leave it. The charged latency includes the real network round trip.
Stateless per turn, like ActionMan: the game's goal + this turn's observation in, one tool call out."""
from __future__ import annotations

import json
import os

from .agents import Agent

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def env_key(name: str) -> str:
    """Read one key from the project .env without printing it."""
    if os.environ.get(name):
        return os.environ[name]
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8-sig"):
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"{name} missing from .env")


class OpenAIAgent(Agent):
    def __init__(self, model: str = "gpt-6-luna", effort: str = "low"):
        from openai import OpenAI
        self.client = OpenAI(api_key=env_key("OPENAI_API_KEY"), timeout=30)
        self.model, self.effort = model, effort
        self.name, self.label = f"openai-{model}", model.upper().replace("GPT-6-", "GPT-6 ")
        self.note = f"API from the laptop, reasoning {effort}; latency includes network"

    def act(self, state, tools, game):
        r = self.client.responses.create(               # gpt-6-luna takes function tools only via /v1/responses
            model=self.model, instructions=game.goal, input=state, reasoning={"effort": self.effort},
            tools=[{"type": "function", **t} for t in tools], tool_choice="required", parallel_tool_calls=False)
        out = []
        for item in r.output:
            if item.type != "function_call":
                continue
            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": item.arguments}
            out.append({"name": item.name, "args": args})
        return out
