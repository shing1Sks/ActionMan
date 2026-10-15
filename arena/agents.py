"""Drivers. Every driver: act(observation, tools, game) -> list of calls [{name, args}]. The runner times it.
`game` carries the world, the level's control scheme, the game's goal text and its perfect-driver plan."""
from __future__ import annotations

import random


class Agent:
    name = "agent"
    label = "Agent"
    note = ""

    def act(self, obs: str, tools: list[dict], game) -> list[dict]:
        raise NotImplementedError


class RandomAgent(Agent):
    name, label, note = "random", "Random", "floor: random tool and arguments"

    def __init__(self, seed: int = 0):
        self.r = random.Random(seed)

    def act(self, obs, tools, game):
        t = self.r.choice(tools)
        args = {}
        for p, spec in t["parameters"]["properties"].items():
            args[p] = self.r.choice(spec["enum"]) if "enum" in spec else self.r.randint(1, 3)
        return [{"name": t["name"], "args": args}]


class ScriptedAgent(Agent):
    """The perfect driver: executes the game's own plan instantly, in this level's names."""
    name, label, note = "scripted", "Perfect driver", "executes the plan instantly"

    def act(self, obs, tools, game):
        return [game.scheme.encode(game.plan(game.world))]
