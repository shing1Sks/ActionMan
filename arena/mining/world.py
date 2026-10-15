"""Mining: dig to a diamond, survive zombies. Deterministic, no I/O; the runner drives it.
The clock is the model: every action costs one 10 ms tick, everything else is the driver's real thinking time."""
from __future__ import annotations

import random
from collections import deque

W, H = 32, 24
SURFACE = 3                      # first ground row; rows above are sky
TPS = 100                        # ticks per second (10 ms)
MAX_TICKS = 120 * TPS
AIR, DIRT, STONE, DIAMOND, BEDROCK = ".", "d", "s", "D", "#"
HARDNESS = {DIRT: 1, STONE: 2, DIAMOND: 2}
NAMES = {AIR: "air", DIRT: "dirt", STONE: "stone", DIAMOND: "diamond", BEDROCK: "bedrock"}
DIRS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
ACT = 1                          # every action, valid or not, costs one tick
MAX_HP, ZOMBIE_HP = 10, 2
ZOMBIE_STEP, ZOMBIE_HIT, SPAWN_EVERY, MAX_ZOMBIES = 50, 100, 3000, 4   # zombies: a block every 0.5 s, 1 damage per s


class World:
    def __init__(self, seed: int = 0, level: int = 0):
        self.seed, self.level = seed, level
        self.rng = random.Random(f"arena-{level}-{seed}")
        self.t = 0
        self.hp, self.food = MAX_HP, 3
        self.outcome = None                       # "win" | "dead" | "timeout"
        self.grid = self._generate()
        self.grid0 = self.grid_rows()
        self.dmg = {}                             # (x, y) -> digs so far
        self.px, self.py = W // 2, SURFACE - 1
        self.zombies = []                         # dicts: x, y, hp, cd
        if level >= 1:
            self._spawn_initial()
        self.frames, self.blocks = [], []
        self._record()

    # ---------- world ----------
    def _generate(self):
        r = self.rng
        g = [[AIR] * W for _ in range(H)]
        for y in range(SURFACE, H):
            for x in range(W):
                deep = (y - SURFACE) / (H - SURFACE)
                g[y][x] = STONE if r.random() < 0.25 + 0.75 * deep else DIRT
        for _ in range(4):                        # caves
            cx, cy, rad = r.randint(4, W - 5), r.randint(SURFACE + 5, H - 5), r.randint(1, 2)
            for y in range(cy - rad, cy + rad + 1):
                for x in range(cx - rad - 1, cx + rad + 2):
                    g[y][x] = AIR
        if self.level == 0:                       # one close diamond, about 8 blocks away
            spots = [(W // 2 + r.randint(-3, 3), SURFACE + r.randint(4, 6))]
        else:
            spots = [(r.randint(2, W - 3), r.randint(H - 5, H - 2)) for _ in range(3)]
        for x, y in spots:
            g[y][x] = DIAMOND
        for x in range(W):
            g[H - 1][x] = BEDROCK
        for y in range(H):
            g[y][0] = g[y][W - 1] = BEDROCK
        return g

    def cell(self, x, y):
        return self.grid[y][x] if 0 <= x < W and 0 <= y < H else BEDROCK

    def zombie_at(self, x, y):
        return next((z for z in self.zombies if z["x"] == x and z["y"] == y), None)

    def _air_cells(self, y0, y1):
        return [(x, y) for y in range(y0, y1) for x in range(1, W - 1) if self.grid[y][x] == AIR]

    def _spawn_initial(self):
        caves = self._air_cells(SURFACE + 1, H - 1)
        for x, y in self.rng.sample(caves, min(2, len(caves))):
            self.zombies.append({"x": x, "y": y, "hp": ZOMBIE_HP, "cd": 0})
        self._spawn_surface()

    def _spawn_surface(self):
        if len(self.zombies) >= MAX_ZOMBIES:
            return
        x = self.rng.choice([2, 3, W - 4, W - 3])
        if not self.zombie_at(x, SURFACE - 1) and (x, SURFACE - 1) != (self.px, self.py):
            self.zombies.append({"x": x, "y": SURFACE - 1, "hp": ZOMBIE_HP, "cd": 0})

    # ---------- time ----------
    def advance(self, ticks: int):
        for _ in range(max(0, ticks)):
            if self.outcome:
                return
            self.t += 1
            if self.zombies and self.t % ZOMBIE_STEP == 0:
                self._move_zombies()
            for z in self.zombies:
                z["cd"] = max(0, z["cd"] - 1)
                if abs(z["x"] - self.px) + abs(z["y"] - self.py) == 1 and z["cd"] == 0:
                    self.hp -= 1
                    z["cd"] = ZOMBIE_HIT
            if self.level >= 1 and self.t % SPAWN_EVERY == 0:
                self._spawn_surface()
            if self.hp <= 0:
                self.hp, self.outcome = 0, "dead"
            elif self.t >= MAX_TICKS:
                self.outcome = "timeout"
            self._record()

    def _move_zombies(self):
        dist = {(self.px, self.py): 0}
        q = deque([(self.px, self.py)])
        while q:                                   # BFS from the player through air
            x, y = q.popleft()
            for dx, dy in DIRS.values():
                n = (x + dx, y + dy)
                if n not in dist and self.cell(*n) == AIR and dist[(x, y)] < 40:
                    dist[n] = dist[(x, y)] + 1
                    q.append(n)
        for z in self.zombies:
            here = dist.get((z["x"], z["y"]))
            if here is None or here <= 1:
                continue
            for dx, dy in DIRS.values():
                n = (z["x"] + dx, z["y"] + dy)
                if dist.get(n, 99) < here and not self.zombie_at(*n) and n != (self.px, self.py):
                    z["x"], z["y"] = n
                    break

    def _record(self):
        f = {"t": self.t, "p": [self.px, self.py], "hp": self.hp, "food": self.food, "z": [[z["x"], z["y"], z["hp"]] for z in self.zombies]}
        if self.frames and all(self.frames[-1][k] == f[k] for k in ("p", "hp", "food", "z")) and not self.outcome:
            return                                 # sparse: only record changes
        self.frames.append(f)

    def replay(self) -> dict:
        return {"grid0": self.grid0, "blocks": self.blocks}

    def summary(self) -> dict:
        return {"hp": self.hp}

    # ---------- actions ----------
    def perform(self, action: tuple | None) -> str:
        """action = (kind, args) canonical, or None for an invalid/empty turn. Returns a short result."""
        if self.outcome:
            return "game over"
        if action is None:
            self.advance(ACT)
            return "wasted turn"
        kind, args = action
        d, steps = args.get("direction"), args.get("steps") or 1
        if kind == "eat":
            if self.food <= 0:
                res = "no food left"
            else:
                self.food -= 1
                self.hp = min(MAX_HP, self.hp + 3)
                res = "ate, health %d" % self.hp
            self.advance(ACT)
            return res
        dx, dy = DIRS[d]
        tx, ty = self.px + dx, self.py + dy
        if kind == "move":
            moved = 0
            for _ in range(steps):
                tx, ty = self.px + dx, self.py + dy
                if self.cell(tx, ty) != AIR or self.zombie_at(tx, ty):
                    break
                self.px, self.py = tx, ty
                moved += 1
            self.advance(ACT)
            if not moved:
                return "blocked by %s" % ("zombie" if self.zombie_at(tx, ty) else NAMES[self.cell(tx, ty)])
            return "moved %d" % moved
        if kind == "attack":
            z = self.zombie_at(tx, ty)
            if z:
                z["hp"] -= 1
                if z["hp"] <= 0:
                    self.zombies.remove(z)
            self.advance(ACT)
            return ("killed zombie" if z["hp"] <= 0 else "hit zombie") if z else "hit nothing"
        # dig
        b = self.cell(tx, ty)
        if b not in HARDNESS:
            self.advance(ACT)
            return "nothing to dig" if b == AIR else "bedrock"
        self.dmg[(tx, ty)] = self.dmg.get((tx, ty), 0) + 1
        if self.dmg[(tx, ty)] < HARDNESS[b]:
            self.advance(ACT)
            return "cracked %s" % NAMES[b]
        self.grid[ty][tx] = AIR
        self.blocks.append({"t": self.t, "x": tx, "y": ty, "b": AIR})
        if b == DIAMOND:
            self.outcome = "win"
            self._record()
            return "DIAMOND!"
        self.advance(ACT)
        return "broke %s" % NAMES[b]

    # ---------- observation ----------
    def nearest_diamond(self):
        ds = [(x, y) for y in range(H) for x in range(W) if self.grid[y][x] == DIAMOND]
        return min(ds, key=lambda c: abs(c[0] - self.px) + abs(c[1] - self.py)) if ds else None

    def state_line(self) -> str:
        parts = [f"Health {self.hp}/{MAX_HP}, food {self.food}."]
        near = [z for z in self.zombies if abs(z["x"] - self.px) + abs(z["y"] - self.py) <= 8]
        if near:
            z = min(near, key=lambda z: abs(z["x"] - self.px) + abs(z["y"] - self.py))
            parts.append(f"Zombie {_rel(z['x'] - self.px, z['y'] - self.py)}.")
        elif self.level >= 1:
            parts.append("No zombie near.")
        d = self.nearest_diamond()
        if d:
            parts.append(f"Diamond {_rel(d[0] - self.px, d[1] - self.py)}.")
        around = []
        for name, (dx, dy) in DIRS.items():
            x, y = self.px + dx, self.py + dy
            around.append(f"{name} {'zombie' if self.zombie_at(x, y) else NAMES[self.cell(x, y)]}")
        parts.append("Around you: " + ", ".join(around) + ".")
        return " ".join(parts)

    def grid_rows(self):
        return ["".join(r) for r in self.grid]


def _rel(dx, dy):
    out = []
    if dy:
        out.append(f"{abs(dy)} {'down' if dy > 0 else 'up'}")
    if dx:
        out.append(f"{abs(dx)} {'right' if dx > 0 else 'left'}")
    return ", ".join(out) or "here"
