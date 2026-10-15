"""Cycling: a short road race where every pedal() call is one stroke, so calls per second is power.
Navigation keeps it honest: the right gear for the terrain, the right branch at forks, brakes before hazards.
Deterministic, no I/O; the runner drives it. Every action costs one 10 ms tick, the rest is the driver's thinking."""
from __future__ import annotations

import random

TPS = 100                        # ticks per second (10 ms)
MAX_TICKS = 90 * TPS
ACT = 1                          # every action, valid or not, costs one tick
VMAX = 20.0                      # m/s a stroke can push you towards (72 km/h)
STROKE = 0.55                    # m/s a stroke adds from standstill; less as you near VMAX (saturates ~30-40 strokes/s)
DRAG = 1 / 300                   # fraction of speed lost per tick (3 s time constant)
GRAV = 20.0                      # m/s^2 per unit grade: climbs slow you, descents speed you up (arcade scale)
EFF = {"low": {"climb": 1.0, "flat": 0.45, "desc": 0.3},     # stroke efficiency by gear and terrain
       "high": {"climb": 0.3, "flat": 1.0, "desc": 1.0}}
SAFE = 6.0                       # m/s through a hazard; faster without brakes = crash
CRASH = 100                      # ticks on the ground after a crash
BRAKE_MISS = 0.6                 # braking with nothing ahead keeps 60% of your speed
LOOK = 8                         # m: the gear should already suit the terrain this far ahead
FORK_WIN, HAZ_WIN = 40, 30       # m: how far ahead a fork / hazard can be prepared for


class World:
    def __init__(self, seed: int = 0, level: int = 0):
        self.seed, self.level = seed, level
        self.rng = random.Random(f"cycling-{level}-{seed}")
        self.t = 0
        self.x, self.v, self.odo = 0.0, 0.0, 0.0  # position on the course (m), speed (m/s), metres actually ridden
        self.gear, self.strokes, self.crashes, self.wrong = "high", 0, 0, 0
        self.stun, self.detour = 0, 0.0           # ticks left on the ground; detour metres left after a wrong branch
        self.turn_to, self.armed = None, False    # chosen branch for the next fork; brakes set for the next hazard
        self.outcome = None                       # "win" | "timeout"
        self.segs, self.forks, self.hazards, self.length = self._generate()
        self.events, self.frames = [], []
        self._record(True)

    # ---------- course ----------
    def _generate(self):
        r, segs, x = self.rng, [], 0
        forks, hazards = [], []

        def seg(n, grade):
            nonlocal x
            segs.append([x, x + n, grade])
            x += n

        if self.level == 0:                       # flat, one fork
            seg(r.randint(75, 85), 0)
            forks.append({"x": r.randint(40, 48), "c": r.choice(["left", "right"]), "d": r.randint(28, 36)})
            return segs, forks, hazards, x
        seg(r.randint(14, 18), 0)
        for _ in range(2):                        # two hills: climb, (top), descent into a hazard, flat with a fork
            seg(r.randint(12, 16), r.randint(5, 9))
            if r.random() < 0.5:
                seg(r.randint(5, 8), 0)
            seg(r.randint(12, 16), -r.randint(5, 9))
            hazards.append({"x": x - 3, "l": 8, "k": r.choice(["gravel", "bend"])})
            n = r.randint(22, 26)
            forks.append({"x": x + n - 8, "c": r.choice(["left", "right"]), "d": r.randint(28, 36)})
            seg(n, 0)
        seg(r.randint(8, 10), 0)
        return segs, forks, hazards, x

    def terrain(self, x: float) -> tuple[str, int]:
        for a, b, g in self.segs:
            if a <= x < b:
                return ("climb" if g > 0 else "desc" if g < 0 else "flat"), g
        return "flat", 0

    def want_gear(self) -> str:
        return "low" if "climb" in (self.terrain(self.x)[0], self.terrain(self.x + LOOK)[0]) else "high"

    def next_fork(self):
        return next((f for f in self.forks if f["x"] > self.x and self.detour <= 0), None)

    def next_hazard(self):
        return next((h for h in self.hazards if h["x"] > self.x), None)

    def in_hazard(self):
        return next((h for h in self.hazards if h["x"] <= self.x < h["x"] + h["l"]), None)

    # ---------- time ----------
    def advance(self, ticks: int):
        for _ in range(max(0, ticks)):
            if self.outcome:
                return
            self.t += 1
            if self.stun:
                self.stun -= 1
            else:
                kind, g = self.terrain(self.x)
                self.v = max(0.0, self.v * (1 - DRAG) - GRAV * g / 100 / TPS)
                if self.in_hazard():
                    self.v = min(self.v, SAFE)
                self._move(self.v / TPS)
            if self.x >= self.length:
                self.x, self.outcome = self.length, "win"
            elif self.t >= MAX_TICKS:
                self.outcome = "timeout"
            self._record(bool(self.outcome))

    def _move(self, dx):
        self.odo += dx
        if self.detour > 0:                       # riding the long way round; rejoin at the fork
            self.detour -= dx
            if self.detour <= 0:
                self.detour = 0.0
                self.events.append({"t": self.t, "k": "rejoin"})
            return
        x0, x1 = self.x, self.x + dx
        for f in self.forks:
            if x0 < f["x"] <= x1:
                ok = self.turn_to == f["c"]
                self.events.append({"t": self.t, "k": "fork" if ok else "wrong", "x": f["x"]})
                self.turn_to = None
                if not ok:
                    self.wrong += 1
                    self.x, self.detour = f["x"], float(f["d"])
                    return
        for h in self.hazards:
            if x0 < h["x"] <= x1:
                if self.armed:
                    self.v = min(self.v, SAFE)
                    self.events.append({"t": self.t, "k": "slow", "x": h["x"]})
                elif self.v > SAFE:
                    self.crashes += 1
                    self.v, self.stun, self.x = 0.0, CRASH, h["x"]
                    self.events.append({"t": self.t, "k": "crash", "x": h["x"]})
                    self.armed = False
                    return
                self.armed = False
        self.x = x1

    def _record(self, force=False):
        f = {"t": self.t, "x": round(self.x, 2), "v": round(self.v, 2), "g": int(self.gear == "high"), "n": self.strokes,
             "o": round(self.odo, 2), "c": int(self.stun > 0), "d": round(self.detour, 1)}
        p = self.frames[-1] if self.frames else None
        if p and not force and f["t"] - p["t"] < 10 and all(p[k] == f[k] for k in ("g", "n", "c")) and (p["d"] > 0) == (f["d"] > 0):
            return                                 # sparse: every 100 ms while coasting, every change otherwise
        self.frames.append(f)

    def replay(self) -> dict:
        return {"len": self.length, "segs": self.segs, "forks": self.forks, "haz": self.hazards, "ev": self.events}

    def summary(self) -> dict:
        return {"strokes": self.strokes, "crashes": self.crashes, "wrong_turns": self.wrong, "m": round(self.x)}

    # ---------- actions ----------
    def perform(self, action: tuple | None) -> str:
        """action = (kind, args) canonical, or None for an invalid/empty turn. Returns a short result."""
        if self.outcome:
            return "game over"
        res = "wasted turn"
        if action is not None:
            kind, args = action
            d = args.get("direction")
            if kind == "pedal":
                if self.stun:
                    res = "still on the ground"
                else:
                    self.strokes += 1
                    eff = EFF[self.gear][self.terrain(self.x)[0]]
                    self.v += STROKE * eff * max(0.0, 1 - self.v / VMAX)
                    res = "%d km/h" % round(self.v * 3.6)
            elif kind == "shift":
                g = "high" if d == "up" else "low"
                res = "already in %s gear" % g if g == self.gear else "%s gear" % g
                self.gear = g
            elif kind == "turn":
                f = self.next_fork()
                if f and f["x"] - self.x <= FORK_WIN:
                    self.turn_to = d
                    res = "taking the %s branch" % d
                else:
                    res = "no fork ahead"
            elif kind == "brake":
                h = self.next_hazard()
                if h and h["x"] - self.x <= HAZ_WIN:
                    self.armed = True
                    res = "brakes set for the %s" % h["k"]
                else:
                    self.v *= BRAKE_MISS
                    res = "braked for nothing"
        self._record(True)
        self.advance(ACT)
        return res
