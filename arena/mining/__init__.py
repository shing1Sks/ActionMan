"""Mining: dig down to a diamond while zombies close in."""
from .observe import Observer, plan
from .world import MAX_TICKS, TPS, World

NAME = "mining"
GOAL = ("You control a miner in a 2D mining game. Reach and mine a diamond, and stay alive. "
        "Each turn you get one line about what is in front of you. Answer with exactly one tool call, as fast as you can.")
_DIR = {"type": "string", "values": "dir", "desc": "Which way."}
SPEC = {
    "tools": {
        "move": {"desc": "Walk through open air in a direction.",
                 "params": {"direction": _DIR, "steps": {"type": "integer", "desc": "How many blocks, 1 to 3.", "optional": True,
                                                         "default": 1, "min": 1, "max": 3}}},
        "dig": {"desc": "Dig out the block in a direction when it blocks your way.", "params": {"direction": _DIR}},
        "attack": {"desc": "Hit a zombie next to you.", "params": {"direction": _DIR}},
        "eat": {"desc": "Eat one food to heal 3 health. Use it when health is low.", "params": {}},
    },
    "tool_aliases": {"move": ["walk", "go", "step"], "dig": ["mine", "excavate", "break_block"],
                     "attack": ["hit", "strike", "fight"], "eat": ["consume", "snack", "heal"]},
    "param_aliases": {"direction": ["dir", "towards", "side"], "steps": ["count", "distance", "n"]},
    "values": {"dir": {"canon": ["left", "right", "up", "down"],
                       "sets": [{"left": "west", "right": "east", "up": "north", "down": "south"},
                                {"left": "L", "right": "R", "up": "U", "down": "D"}]}},
}
LEVELS = ["Warm-up: no zombies, the diamond is close", "Survive: zombies hunt you",
          "Renamed tools: new tool and parameter names", "Renamed values: directions renamed too"]
