"""Cycling: every pedal() call is one stroke, so the faster the driver, the faster the bike."""
from .observe import Observer, plan
from .world import MAX_TICKS, TPS, World

NAME = "cycling"
GOAL = ("You ride a bike in a road race. Reach the finish line as fast as you can. Each turn you get one line about the road "
        "ahead. Answer with exactly one tool call, as fast as you can: every pedal stroke is one call.")
SPEC = {
    "tools": {
        "pedal": {"desc": "Push one pedal stroke. Every stroke adds speed, so pedal whenever nothing else needs doing.", "params": {}},
        "shift": {"desc": "Change gear: a low gear for climbs, a high gear everywhere else. The wrong gear wastes your strokes.",
                  "params": {"direction": {"type": "string", "values": "gear", "desc": "down for a lower gear, up for a higher gear."}}},
        "turn": {"desc": "Take a branch at a fork in the road. The wrong branch is a long detour.",
                 "params": {"direction": {"type": "string", "values": "side", "desc": "The side the finish is on."}}},
        "brake": {"desc": "Brake for a hazard ahead (gravel, a sharp bend) so you reach it at a safe speed. "
                          "Hitting a hazard fast without braking crashes you.", "params": {}},
    },
    "tool_aliases": {"pedal": ["stroke", "push", "crank"], "shift": ["shifter", "derailleur", "cog"],
                     "turn": ["steer", "take_branch", "fork"], "brake": ["slow_down", "squeeze_brakes", "check_speed"]},
    "param_aliases": {"direction": ["way", "dir", "towards"]},
    "values": {"side": {"canon": ["left", "right"], "sets": [{"left": "L", "right": "R"}, {"left": "west", "right": "east"}]},
               "gear": {"canon": ["up", "down"], "sets": [{"up": "higher", "down": "lower"}, {"up": "plus", "down": "minus"}]}},
}
LEVELS = ["Warm-up: a flat road with one fork", "Full course: climbs, descents, hazards and forks",
          "Renamed tools: new tool and parameter names", "Renamed values: directions and gears renamed too"]
