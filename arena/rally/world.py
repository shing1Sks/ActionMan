"""Rally: a point-to-point special stage driven from pace notes. Deterministic, no I/O; the runner drives it.
The car never stops: every 10 ms tick it covers v/TPS metres. Each corner is checked at its entry: the called line
must match it (direction, severity within 1) and a hairpin must be taken at hairpin speed, else the car crashes."""
from __future__ import annotations

import math
import random

TPS = 100                        # ticks per second (10 ms)
MAX_TICKS = 120 * TPS
ACT = 1                          # every action, valid or not, costs one tick
V_START, V_HAIR, V_RESTART = 25.0, 25.0, 15.0     # m/s: rolling start, hairpin limit, after a crash
ACC, DEC = 55.0, 120.0           # m/s^2
MARGIN = 35.0                    # m: the brake note comes this far before the last braking point
HAIR_ZONE = 160.0                # m: no throttle call this close to a hairpin
CRASH_TICKS, MAX_CRASHES = 150, 5
ANGLE = {1: 175, 2: 110, 3: 82, 4: 58, 5: 38, 6: 20}   # degrees turned per severity (1 = hairpin, 6 = kink)
ARC = {1: 44, 2: 30, 3: 28, 4: 26, 5: 24, 6: 22}       # metres of road per corner
STEP = 2.0                                             # centerline sample spacing, m
CFG = {  # level 0: slow, few corners, generous gaps. Level 1+: the full stage.
    0: dict(vmax=70.0, single=3, complex=0, hairpin=1, long=0, gap=(90, 130), sev=(3, 6)),
    1: dict(vmax=100.0, single=4, complex=2, hairpin=1, long=1, gap=(40, 95), sev=(2, 6)),
}


class Corner:
    __slots__ = ("s", "dir", "sev", "len")

    def __init__(self, s, d, sev):
        self.s, self.dir, self.sev, self.len = s, d, sev, ARC[sev]


class World:
    def __init__(self, seed: int = 0, level: int = 0):
        self.seed, self.level = seed, level
        self.rng = random.Random(f"rally-{level}-{seed}")
        self.cfg = CFG[min(level, 1)]
        self.vmax = self.cfg["vmax"]
        for _ in range(60):                       # redraw until the road never crosses itself
            self.corners, self.length, self.pts = self._generate()
            if self._clear():
                break
        self.t, self.s, self.v, self.target = 0, 0.0, V_START, V_START
        self.line, self.line_s = None, 0.0        # the called line: (direction, severity), and where it was called
        self.ci, self.stop, self.crashes = 0, 0, 0
        self.outcome = None                       # "win" | "dead" | "timeout"
        self.events, self.frames = [], []
        self._record()

    # ---------- stage ----------
    def _generate(self):
        r, c = self.rng, self.cfg
        feats = ["single"] * c["single"] + ["complex"] * c["complex"] + ["hairpin"] * c["hairpin"] + ["long"] * c["long"]
        r.shuffle(feats)
        segs, head = [("S", r.uniform(70, 90))], 0.0          # head: running heading in degrees, 0 = east

        def corner(sev, d=None):
            nonlocal head
            if d is None:                          # steer the stage back toward east so it runs point to point
                d = "right" if head > 50 else "left" if head < -50 else r.choice(["left", "right"])
            head += ANGLE[sev] * (1 if d == "left" else -1)
            segs.append(("C", d, sev))
            return d

        def sev():
            return r.randint(*c["sev"])

        for f in feats:
            if f == "single":
                segs.append(("S", r.uniform(*c["gap"])))
                corner(sev())
            elif f == "long":
                segs.append(("S", r.uniform(150, 180)))
                corner(r.randint(4, 6))
            elif f == "hairpin":
                segs.append(("S", r.uniform(140, 170)))
                corner(1, "left" if head <= 0 else "right")
                segs.append(("S", r.uniform(45, 60)))
                corner(r.randint(2, 4))
            else:                                  # a complex: 2-3 corners back to back, alternating
                segs.append(("S", r.uniform(*c["gap"])))
                d = corner(r.randint(3, 6))
                for _ in range(r.randint(1, 2)):
                    segs.append(("S", r.uniform(4, 16)))
                    d = corner(r.randint(3, 6), "left" if d == "right" else "right")
        segs.append(("S", r.uniform(80, 100)))
        corners, pts, s, x, y, h = [], [[0.0, 0.0]], 0.0, 0.0, 0.0, 0.0
        for seg in segs:                           # integrate the centerline every STEP metres (math axes: y up)
            if seg[0] == "S":
                n, dh, ln = max(1, round(seg[1] / STEP)), 0.0, seg[1]
            else:
                cn = Corner(s, seg[1], seg[2])
                corners.append(cn)
                ln = cn.len
                n = round(ln / STEP)
                dh = math.radians(ANGLE[seg[2]]) / n * (1 if seg[1] == "left" else -1)
            for _ in range(n):
                h += dh / 2
                x, y = x + math.cos(h) * ln / n, y + math.sin(h) * ln / n
                h += dh / 2
                pts.append([x, y])
            s += ln
        return corners, s, pts

    def _clear(self, gap=26.0, skip=60):
        """True if no two stretches of road more than `skip` samples apart come within `gap` metres."""
        P = self.pts
        for i in range(0, len(P), 2):
            for j in range(i + skip, len(P), 2):
                if (P[i][0] - P[j][0]) ** 2 + (P[i][1] - P[j][1]) ** 2 < gap * gap:
                    return False
        return True

    def next_corner(self):
        return self.corners[self.ci] if self.ci < len(self.corners) else None

    def brake_point(self) -> float:
        """Distance before a hairpin where the brake call is due at the current speed."""
        v = max(self.v, self.target)
        return max(0.0, v * v - V_HAIR * V_HAIR) / (2 * DEC) + MARGIN

    # ---------- time ----------
    def advance(self, ticks: int):
        for _ in range(max(0, ticks)):
            if self.outcome:
                return
            self.t += 1
            if self.stop:
                self.stop -= 1
                if not self.stop:
                    self.v = V_RESTART
            else:
                c, goal = self.next_corner(), self.target
                if goal == V_HAIR and c and c.sev == 1 and c.s - self.s > (self.v ** 2 - V_HAIR ** 2) / (2 * DEC) + 2:
                    goal = max(goal, self.v)           # brake called for a hairpin: hold speed to the last braking point
                step = (ACC if self.v < goal else DEC) / TPS
                self.v = min(goal, self.v + step) if self.v < goal else max(goal, self.v - step)
                self.s += self.v / TPS
                if c and self.s >= c.s:
                    self._enter(c)
                if self.s >= self.length:
                    self.s, self.outcome = self.length, "win"
            if not self.outcome and self.t >= MAX_TICKS:
                self.outcome = "timeout"
            self._record()

    def _enter(self, c):
        ok_line = bool(self.line) and self.line[0] == c.dir and abs(self.line[1] - c.sev) <= 1
        ok_speed = c.sev > 1 or self.v <= V_HAIR + 1
        ev = {"t": self.t, "c": self.ci}
        if ok_line and ok_speed:
            ev["m"] = round(c.s - self.line_s)     # how far ahead the line was called (the viewer draws late calls ragged)
        else:
            ev["crash"] = "line" if not ok_line else "speed"
            self.crashes += 1
            self.s, self.v, self.target, self.stop = c.s, 0.0, V_RESTART, CRASH_TICKS
            if self.crashes >= MAX_CRASHES:
                self.outcome = "dead"
        self.events.append(ev)
        self.ci += 1
        self.line = None

    def line_code(self):
        return f"{self.line[0][0].upper()}{self.line[1]}" if self.line else ""

    def _record(self):
        f = {"t": self.t, "s": round(self.s, 2), "v": round(self.v, 1), "ln": self.line_code(), "cr": self.crashes, "x": int(self.stop > 0)}
        if self.frames and self.t % 5 and not self.outcome and all(self.frames[-1][k] == f[k] for k in ("ln", "cr", "x")):
            return                                 # sparse: every 50 ms, plus every change of line or crash state
        self.frames.append(f)

    def replay(self) -> dict:
        return {"len": round(self.length, 1), "vmax": self.vmax, "pts": [[round(x, 1), round(y, 1)] for x, y in self.pts],
                "corners": [{"s": round(c.s, 1), "dir": c.dir, "sev": c.sev, "len": c.len} for c in self.corners],
                "events": self.events}

    def summary(self) -> dict:
        return {"crashes": self.crashes}

    # ---------- actions ----------
    def perform(self, action: tuple | None) -> str:
        """action = (kind, args) canonical, or None for an invalid/empty turn. Returns a short result."""
        if self.outcome:
            return "game over"
        if action is None:
            self.advance(ACT)
            return "wasted turn"
        kind, args = action
        if kind == "steer":
            line = (args["direction"], args["severity"])
            if line != self.line:
                self.line, self.line_s = line, self.s
            res = f"line {line[0]} {line[1]}"
        elif kind == "brake":
            self.target = min(self.target, V_HAIR)
            res = "braking"
        else:
            self.target = self.vmax
            res = "flat out"
        self.advance(ACT)
        return res
