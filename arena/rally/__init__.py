"""Rally: drive a special stage from the co-driver's pace notes at full speed."""
from .observe import Observer, plan
from .world import MAX_TICKS, TPS, World

NAME = "rally"
GOAL = ("You drive a rally car on a special stage. Each turn your co-driver reads one pace note about the road ahead. "
        "Answer with exactly one tool call, as fast as you can: the car keeps going while you think.")
_DIR = {"type": "string", "values": "dir", "desc": "Which way the corner turns."}
SPEC = {
    "tools": {
        "steer": {"desc": "Turn into the corner the co-driver calls: its direction and its severity number, 1 to 6.",
                  "params": {"direction": _DIR, "severity": {"type": "integer", "desc": "Corner severity from the note, 1 to 6.",
                                                             "min": 1, "max": 6}}},
        "brake": {"desc": "Brake hard. Only when the co-driver says brake.", "params": {}},
        "throttle": {"desc": "Full throttle. Only when the co-driver says flat out or full throttle.", "params": {}},
    },
    "tool_aliases": {"steer": ["turn", "corner", "set_line"], "brake": ["slow_down", "anchor", "brakes"],
                     "throttle": ["accelerate", "gas", "full_power"]},
    "param_aliases": {"direction": ["side", "way", "dir"], "severity": ["grade", "sharpness", "number"]},
    "values": {"dir": {"canon": ["left", "right"],
                       "sets": [{"left": "port", "right": "starboard"}, {"left": "L", "right": "R"}]}},
}
LEVELS = ["Warm-up: a short, slow stage with wide gaps", "Full stage: flat-out complexes and hairpins",
          "Renamed tools: new tool and parameter names", "Renamed values: left and right renamed too"]
