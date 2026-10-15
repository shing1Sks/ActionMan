"""Cycling: the perfect driver's plan, and what a driver is told each turn.
The observation states the situation that makes the call obvious under the tool rules ("Fork in 40 m: the finish
is left.", "Climb of 8% in 6 m, you need a lower gear."), never the order. Plain left/right/up/down words, digits for numbers."""
from __future__ import annotations

import random

from .world import FORK_WIN, HAZ_WIN, SAFE

TURN = ["Fork in {d} m: the finish is {side}.", "The road splits in {d} m, the finish is to the {side}.",
        "Fork in {d} m, the finish line is {side}."]
BRAKE = ["Hazard ahead: {haz} in {d} m, {kmh} km/h.", "Hazard: {haz} in {d} m at {kmh} km/h.", "Hazard ahead: {haz} {d} m away."]
DOWN = ["Climb of {g}% in {d} m, you need a lower gear.", "A {g}% climb starts in {d} m and the gear has to go down."]
DOWN_ON = ["Climbing {g}% and you need a lower gear.", "On a {g}% climb, the gear has to go down."]
UP = ["Top of the climb, {road} ahead: the gear has to go up.", "Over the top: {road} ahead, the gear has to go up."]
PEDAL = {
    "flat": ["Flat road, {kmh} km/h, {left} m to the finish.", "{left} m to go on a flat road, {kmh} km/h.",
             "Open road at {kmh} km/h, {left} m to the finish."],
    "climb": ["Uphill {g}%, {kmh} km/h, {left} m to go.", "{left} m to go, uphill {g}% at {kmh} km/h."],
    "desc": ["Downhill {g}%, {left} m to go.", "Descending {g}% at {kmh} km/h."],
    "down": ["You crashed and are getting back on the bike.", "Back up after the crash, {left} m to go."],
    "detour": ["{det} m of side road left before the course, {kmh} km/h.", "Still {det} m of side road to the course, {kmh} km/h."],
}
HAZ = {"gravel": "gravel patch", "bend": "sharp bend"}


def plan(w) -> tuple:
    """Best canonical action (kind, args): brakes for a hazard if fast, the finish branch at a fork, the gear for the terrain, else pedal."""
    if w.stun:
        return ("pedal", {})
    h = w.next_hazard()
    if h and h["x"] - w.x <= HAZ_WIN and not w.armed and w.v > SAFE - 1:
        return ("brake", {})
    f = w.next_fork()
    if f and f["x"] - w.x <= FORK_WIN and w.turn_to != f["c"]:
        return ("turn", {"direction": f["c"]})
    g = w.want_gear()
    if g != w.gear:
        return ("shift", {"direction": "down" if g == "low" else "up"})
    return ("pedal", {})


class Observer:
    def __init__(self, seed: int = 0, style: str = "state"):
        self.r = random.Random(f"cycling-say-{seed}")

    def observe(self, w) -> str:
        kind, args = plan(w)
        here, g = w.terrain(w.x)
        v = dict(kmh=round(w.v * 3.6), left=max(1, round(w.length - w.x)), g=abs(g), det=max(1, round(w.detour)))
        if kind == "turn":
            f = w.next_fork()
            text = self.r.choice(TURN).format(d=max(1, round(f["x"] - w.x)), side=f["c"], **{**v, "det": f["d"]})
        elif kind == "brake":
            h = w.next_hazard()
            text = self.r.choice(BRAKE).format(d=max(1, round(h["x"] - w.x)), haz=HAZ[h["k"]], **v)
        elif kind == "shift" and args["direction"] == "down":
            if here == "climb":
                text = self.r.choice(DOWN_ON).format(**v)
            else:
                a = next(s for s in w.segs if s[0] > w.x and s[2] > 0)
                text = self.r.choice(DOWN).format(**{**v, "d": max(1, round(a[0] - w.x)), "g": a[2]})
        elif kind == "shift":
            road = "flat road" if here == "flat" else f"a {abs(g)}% descent"
            text = self.r.choice(UP).format(road=road, **v)
        else:
            state = "down" if w.stun else "detour" if w.detour > 0 else here
            text = self.r.choice(PEDAL[state]).format(**v)
        return text
