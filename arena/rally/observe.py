"""Rally: the perfect driver's plan, and the co-driver's pace note each turn.
The note describes what is ahead in rally style with digits ("Next corner: left 4.", "Brake! Hairpin in 80.",
"Long straight, 300, flat out."), so the call is obvious under the tool rules. Corner notes carry one number,
the severity, and name the corner; finish notes carry none: ActionMan copied stray numbers into severity. Plain left/right always,
so on L3 the driver must translate to the level's values."""
from __future__ import annotations

import random

from .world import HAIR_ZONE, V_HAIR

STEER = ["Next corner: {dir} {n}.", "Corner {dir} {n} next.", "Caution, {dir} {n} corner.", "{D} {n} corner, coming up.", "{D} {n} is the next corner."]
HAIRPIN = ["Hairpin {dir} 1.", "{D} 1 hairpin next.", "Next: hairpin {dir} 1."]
BRAKE = ["Brake! Hairpin in {d}.", "Hairpin in {d}, brake now.", "Brake hard, hairpin {d} ahead.", "Brake, brake! {d} to the hairpin."]
THROTTLE = ["Flat out, {d} to the next corner.", "Full throttle, {d} clear.", "Clear road for {d}, flat out."]
LONG = ["Long straight, {d}, flat out."] + THROTTLE
FINISH = ["Flat out to the finish!", "Finish line ahead, full throttle.", "Last straight, flat out."]


def plan(w) -> tuple:
    """Best canonical action (kind, args): brake into a hairpin, call the next corner, get back on the throttle,
    else hold the called line."""
    c = w.next_corner()
    if c is None:
        return ("throttle", {})
    d, line = c.s - w.s, ("steer", {"direction": c.dir, "severity": c.sev})
    if c.sev == 1 and w.target > V_HAIR and d <= w.brake_point():
        return ("brake", {})
    if w.line != (c.dir, c.sev):
        return line
    if w.target < w.vmax and not (c.sev == 1 and d <= HAIR_ZONE):
        return ("throttle", {})
    return line


def _dist(m):
    return str(max(10, int(round(m / 10.0)) * 10))


class Observer:
    def __init__(self, seed: int = 0):
        self.r = random.Random(f"rally-say-{seed}")

    def observe(self, w) -> str:
        kind, args = plan(w)
        c = w.next_corner()
        d = (c.s if c else w.length) - w.s
        if kind == "steer":
            text = self.r.choice(HAIRPIN if args["severity"] == 1 else STEER)
        elif kind == "brake":
            text = self.r.choice(BRAKE)
        else:
            text = self.r.choice((LONG if d >= 150 else THROTTLE) if c else FINISH)
        dr = args.get("direction", "")
        return text.format(D=dr.capitalize(), dir=dr, n=args.get("severity", ""), d=_dist(d))
