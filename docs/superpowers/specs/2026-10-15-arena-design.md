# ActionMan Arena: design (2026-10-15)

> **Final state (end of 2026-10-15).** This spec records how the arena was designed, starting with mining. What shipped differs:
> - **Three events,** each in its own package: `arena/mining/`, `arena/rally/` and `arena/cycling/`. They share the `Scheme` renaming (`arena/controls.py`) and one viewer shell.
> - **The clock:** it runs at 100 ticks/s. Every action costs one 10 ms tick, and thinking time is charged with a fractional carry, so fast calls are never rounded to zero.
> - **No co-driver:** the "co-driver" is a scripted planner, and the model is never told about it. It only sees one instruction or note per turn.
> - **Claude:** not run.
> - **Results and rules:** `docs/arena/README.md` and the per-event docs.

A showcase, not a rigorous benchmark. Rigorous numbers come from the benchmark plan against base models.
Agents play a tiny 2D Minecraft through parameterised tool calls. The goal is to reach a diamond first and survive zombies.
Nothing is committed; the user reviews.

## Decisions so far

- **Approach A:** latency-charged turns plus a replay viewer.
  - The world keeps moving while a model thinks.
  - A slow model acts on a stale view.
- **Out of the box first:** ActionMan plays with no game training. A game-trained version comes only if wanted later.
- **Contestants:**
  - Now: ActionMan, Qwen3-0.6B and GPT-6 Luna.
  - Later: Claude, once the Anthropic key is added.
  - Always: Scripted (upper bound) and Random (floor).
- **Simplify if it doesn't work:** if models fail even the easy level, the game gets simpler. It does not get more complex.
- **Watch it live:** the user watches the first slice in the browser and gives feedback before the AI players are wired in.
- **Race only (user, after the first slice).** Free mode was tried and dropped.
  - In Free mode the model sees the state line and must decide the move. ActionMan can't plan.
  - Phrasing the state as a request ("I need to reach the diamond. {state} Make my next move.") made it call every time, but it matched the bot's move only 4 of 15 times.
  - GPT-6 Luna planned well but died to zombies at about 1.5 s per move.
- **The Race:** a co-driver plans each move and calls the order out in varied phrasing. Every driver must execute it on the level's switched controls, with real latency charged.
- **Must-call:** ActionMan runs with `require=True` (new opt-in flag in `predict`), the same as `tool_choice="required"` for the API models. Default eval behaviour is unchanged.
- **Make it work first, then make it beautiful:** the viewer is the showcase, so it gets a full visual pass once the race works end to end.

## 1. The game

**World:**
- A 2D cross-section, 32 wide by 24 deep, with no gravity.
- Blocks: air, dirt (1 dig), stone (2 digs), diamond (2 digs), bedrock (border, unbreakable).
- 3 diamonds sit near the bottom, plus a few air caves.
- A fixed seed per level, so every contestant gets the identical world.

**Player:**
- Starts in the sky row above the surface, with 10 health and 3 food.
- Can only move into air.

**Zombies:**
- They spawn in caves and on the surface.
- Every 0.5 s they step one block toward the player through air (shortest path).
- Next to the player, a zombie does 1 damage per second. Two hits kill it.

**Clock:**
- The game runs at 10 ticks/s.
- A turn is: observe, then think (the model's real latency), then the action lands, then the action takes its own time.
- Action times: move 2 ticks per step; dig 3 ticks; attack 3 ticks; eat 5 ticks.
- This way even a 0 ms bot can't act infinitely fast.

**End and score:**
- The game ends on a diamond mined (win), 0 health (dead), or 120 s (timeout).
- Score order: win first, then time to diamond, then health left.

**The co-driver's order**, one per turn, in varied phrasing and always in plain left/right/up/down:
> "Dig down." / "Break the block below you." / "Zombie to your left, hit it!" / "Go right three steps." / "Eat now."

It is planned by the scripted bot from the true world state (`arena/navigator.py`). Wrong or slow execution costs time and health; the co-driver re-plans from wherever the miner ends up.

**Tools** (canonical):
- `move(direction, steps)`: direction is left/right/up/down; steps 1-3.
- `dig(direction)`
- `attack(direction)`
- `eat()`

An invalid call wastes the turn (2 ticks) and is counted.

## 2. Difficulty dial and control switching

| Level | Zombies | Controls |
|---|---|---|
| L0 walk | none | fixed; diamond about 8 blocks away |
| L1 survive | yes | fixed |
| L2 renamed | yes | tool and parameter names renamed per seed (`dig` -> `mine`/`excavate`, `direction` -> `dir`/`towards`) |
| L3 translated | yes | values renamed too (`left` -> `west`/`L`); the state line still says "left" |

We only step up a level once models handle the level below.

## 3. Structure (`arena/`, separate from model code)

| Unit | Job |
|---|---|
| `arena/game.py` | Pure, deterministic engine: `World(seed, level)`, `tick()`, `apply(action)`, `state_line()` |
| `arena/controls.py` | Per-level control scheme: renamed tool schemas, mapping a call back to a canonical action, flagging invalid calls |
| `arena/navigator.py` | The co-driver: the scripted plan, phrased as a natural-language order |
| `arena/agents.py`, `am_agent.py`, `qwen_agent.py`, `api_agents.py` | `act(order, tools)` drivers: Scripted, Random, ActionMan, Qwen3-0.6B SFT, OpenAI (and later Claude). The runner times every call |
| `arena/run.py` | Match runner: same seeds for every agent, real latency charged, one replay JSON per game, plus `viewer/replays.js` |
| `arena/viewer/index.html` | Static split-screen replay: every agent on the same seed, play/pause/speed/scrub, health, a "thinking" bar, and a live call log |

Where things run:
- **Showcase numbers:** GPU models (ActionMan, Qwen) run on a Vast box.
- **First look:** ActionMan runs on the laptop CPU (slower than its real speed; labelled as such).
- **API models:** run from the laptop. The key never leaves it, and the network time is charged.

## 4. Viewer, testing, budget

**Viewer:**
- One canvas per agent, side by side, in sync on game time.
- Each panel shows a health bar, food, a status badge (won at X s / dead / timeout) and a thinking bar while the model's latency runs.
- A scrolling call log shows each call, with invalid calls in red.

**Tests:**
- Engine determinism: same seed gives the same world and the same replay.
- The scripted agent wins every level seed (proves each level is winnable).
- Control translation round-trips at every level.
- Invalid calls are counted and waste the turn.

**Budget:**
- Laptop runs are $0.
- GPT-6 Luna: about 40-80 calls per game, cents per match set.
- A GPU box for ActionMan/Qwen showcase runs: about $0.20 per session.
