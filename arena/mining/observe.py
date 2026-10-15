"""Mining: the perfect driver's plan, and what a driver is told each turn.
style "order": the plan phrased as an order ("Dig down."). style "state": the facts that make the move obvious
under the tool rules ("The way to the diamond is down. Below you: stone."). Plain left/right/up/down either way,
so on L3 the driver must translate to the level's values."""
from __future__ import annotations

import heapq
import random

from .world import AIR, BEDROCK, DIRS, HARDNESS, NAMES, H, W

REL = {"left": "to your left", "right": "to your right", "up": "above you", "down": "below you"}
NUM = {1: ["1", "one"], 2: ["2", "two"], 3: ["3", "three"]}
ORDER = {
    "move": ["Move {d} {n}.", "Go {d} {n} blocks.", "Walk {n} {d}.", "Head {d}, {n} steps.", "{D} {n}, go!"],
    "dig": ["Dig {d}.", "Break the block {rel}.", "Mine {d}!", "Clear the {block} {rel}.", "Dig {rel}, quick."],
    "attack": ["Zombie {rel}, hit it!", "Attack {d}!", "Strike the zombie {rel}.", "Hit {d}, now!"],
    "eat": ["Eat something.", "You're hurt, eat!", "Heal up, eat some food.", "Eat now."],
}
STATE = {
    "move": ["The way {d} is open: {n} free blocks toward the diamond.", "Open air {rel}, {n} blocks toward the diamond."],
    "dig": ["{Block} blocks the way {d}.", "The way to the diamond is {d}, blocked by {block}."],
    "attack": ["Zombie {rel}!", "A zombie is right {rel}."],
    "eat": ["Health {hp}/10, food {food}.", "Health is low: {hp}/10."],
}


def _dir(x0, y0, x1, y1):
    return next(k for k, v in DIRS.items() if v == (x1 - x0, y1 - y0))


def _path(w):
    start = (w.px, w.py)
    cost, prev, pq = {start: 0}, {}, [(0, start)]
    while pq:
        c, (x, y) = heapq.heappop(pq)
        if w.cell(x, y) == "D":
            node, out = (x, y), []
            while node != start:
                out.append(node)
                node = prev[node]
            return out[::-1]
        if c > cost[(x, y)]:
            continue
        for dx, dy in DIRS.values():
            n = (x + dx, y + dy)
            b = w.cell(*n)
            if b == BEDROCK or not (0 <= n[0] < W and 0 <= n[1] < H):
                continue
            nc = c + 2 + 3 * HARDNESS.get(b, 0)
            if nc < cost.get(n, 1e9):
                cost[n], prev[n] = nc, (x, y)
                heapq.heappush(pq, (nc, n))
    return None


def plan(w) -> tuple:
    """Best canonical action (kind, args): fight an adjacent zombie, eat when low, else follow the cheapest dig path."""
    for d, (dx, dy) in DIRS.items():
        if w.zombie_at(w.px + dx, w.py + dy):
            return ("attack", {"direction": d})
    if w.hp <= 5 and w.food > 0:
        return ("eat", {})
    path = _path(w)
    if not path:
        return ("dig", {"direction": "down"})
    (nx, ny), d = path[0], _dir(w.px, w.py, *path[0])
    if w.cell(nx, ny) != AIR:
        return ("dig", {"direction": d})
    steps = 1
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        if steps == 3 or _dir(ax, ay, bx, by) != d or w.cell(bx, by) != AIR:
            break
        steps += 1
    return ("move", {"direction": d, "steps": steps})


class Observer:
    def __init__(self, seed: int = 0, style: str = "order"):        # "order" plays far better (see docs/arena/mining.md)
        self.r, self.style = random.Random(f"mining-say-{seed}"), style

    def observe(self, w) -> str:
        kind, args = plan(w)
        text = self.r.choice((ORDER if self.style == "order" else STATE)[kind])
        d, steps = args.get("direction"), args.get("steps", 1)
        block = ""
        if d:
            dx, dy = DIRS[d]
            block = NAMES[w.cell(w.px + dx, w.py + dy)]
        text = text.format(d=d, D=(d or "").capitalize(), rel=REL.get(d, ""), Rel=REL.get(d, "").capitalize(),
                           n=self.r.choice(NUM.get(steps, ["1"])), block=block, Block=block.capitalize(), hp=w.hp, food=w.food)
        if steps == 1:
            text = text.replace(" blocks", " block").replace(" steps", " step")
        return text
