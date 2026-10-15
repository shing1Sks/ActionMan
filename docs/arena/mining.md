# Arena event: 2D Minecraft

Dig down to a diamond while zombies close in. Pixel art, lantern light, follow camera (`arena/viewer/games/mining.js`).

## Rules

- **World:** a 32 x 24 cross-section: sky, dirt (1 dig), stone and diamond (2 digs), bedrock border, a few caves.
  3 diamonds near the bottom, one close diamond on L0. The world comes from a fixed seed, so every model gets the identical map.
- **Player:** 10 health, 3 food.
- **Zombies:** they step one block toward the miner every 0.5 s through open air, and deal 1 damage per second when adjacent.
  Two hits kill a zombie.
- **Tools:** `move(direction, steps 1-3)`, `dig(direction)`, `attack(direction)`, `eat()`. The rules are in the tool descriptions.
- **The clock is the model:** every action costs one 10 ms tick; everything else is the model's real thinking time
  (`arena/run.py`, fractional time carried so fast calls are never rounded away).
- **End:** win on mining a diamond; dead at 0 health; timeout at 120 s.

| Level | What changes |
|---|---|
| L0 Warm-up | no zombies, diamond about 8 blocks away |
| L1 Survive | zombies |
| L2 Renamed tools | tool and parameter names renamed per seed (`dig` -> `mine`, `direction` -> `dir`) |
| L3 Renamed values | directions renamed too (`down` -> `D` or `south`), the observation still says down |

## What the model sees: orders, not state (measured)

Two observation styles were tried, both produced by the same planner:

- **order:** "Dig down." / "Zombie to your left, hit it!"
- **state:** "Stone blocks the way down." / "The way down is open: 2 free blocks toward the diamond."

| 12 races each (L0-L3 x seeds 0-2), RTX 4090 | order | state |
|---|---|---|
| ActionMan (p50 ~12 ms per call) | **10 / 12 won**, 0.3-1.3 s | 1 / 12 won (loops on valid but wrong moves) |
| Qwen3-0.6B SFT (p50 ~175 ms) | 9 / 12 won, 2.6-11.8 s | 8 / 12 won |
| GPT-6 Luna, API from the laptop (p50 ~1.3 s) | 10 / 12 won, 17-70 s, 0 invalid calls | not run |

From state, ActionMan cannot decide between `dig` and `move`: that is a planning step, and ActionMan does not plan.
From orders it is a translation, and it is fast and mostly right. The event uses orders. A 24-moment CPU probe agreed in
direction (right move 13/24 from orders, 10-11/24 from state).

Both models lose L3 for different reasons:
- Qwen sends every call with the plain value (`down` instead of `D`), so every call is invalid.
- ActionMan maps most values but sometimes picks the wrong one (`mine(dir='U')` for "below you") and loops.

## Headline

The style comparison above ran on one box; the final runs for all three events ran on another (`arena-final`, `README.md`).
On the final box ActionMan wins 10 of 12 in 0.4-2.1 s (p50 26 ms per call, a slower host) and Qwen wins 9 of 12 in 4.9-19.1 s.
GPT-6 Luna also wins 10 of 12, in 17-70 s. It never sends an invalid call, even on renamed controls; it loses only to zombies while it thinks.
