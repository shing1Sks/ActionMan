# Arena event: Cycling

Every `pedal()` call is one pedal stroke, so calls per second is power: the faster the driver thinks, the faster the bike.
Navigation keeps it from being a spam test. Flat-vector side view at golden hour (`arena/viewer/games/cycling.js`).

## Rules

- **Course:** generated per (level, seed). L0 is about 80 m of flat road with one fork. L1-L3 are about 125-145 m:
  a start flat, then two hills. Each hill is a climb (5-9%), sometimes a short top, a descent (5-9%) ending in a hazard
  (gravel or a sharp bend), and a flat with a fork. Every model gets the identical course.
- **Tools:** the rules are in the tool descriptions.
  - `pedal()`: one stroke, adds speed with diminishing returns towards 72 km/h.
  - `shift(direction)` up/down: low gear for climbs, high gear everywhere else. The wrong gear cuts stroke efficiency
    to 0.3-0.45.
  - `turn(direction)` left/right: picks the branch for the next fork, within 40 m. The wrong branch is a 28-36 m detour.
  - `brake()`: sets the brakes for the next hazard, within 30 m. Reaching a hazard faster than 22 km/h without brakes
    is a crash: speed 0 and 1 s on the ground. Braking with nothing ahead costs 40% of your speed.
- **Physics** (`arena/cycling/world.py`):
  - Each tick, speed decays with linear drag (3 s time constant) and grade × 20 m/s² (arcade gravity).
  - A stroke adds `0.55 × eff × (1 - v/20)` m/s.
  - Steady speed on the flat in the right gear is about `20 × c / (c + 12)` m/s at `c` strokes/s. That saturates
    around 30-40 strokes/s: 22 strokes/s gives about 47 km/h, 100 strokes/s about 64 km/h.
- **The clock is the model:** every action costs one 10 ms tick, everything else is real thinking time.
- **End:** win at the finish line; timeout at 90 s. `dead` is unused (a crash costs time, not the race).

| Level | What changes |
|---|---|
| L0 Warm-up | flat, one fork, no gears or hazards to handle |
| L1 Full course | climbs, descents, two hazards, two forks |
| L2 Renamed tools | `pedal` -> `stroke`/`push`/`crank`, `shift` -> `shifter`/`derailleur`/`cog`, `turn` -> `steer`/`take_branch`/`fork`, `brake` -> `slow_down`/..., `direction` -> `way`/`dir`/`towards` |
| L3 Renamed values | sides `L`/`R` or `west`/`east`; gears `higher`/`lower` or `plus`/`minus`. The observation still says left/right/up/down |

## What the model sees: the situation

The observation describes the situation and never gives the order. It always uses digits:
"Fork in 20 m: the finish is left." / "Hazard ahead: gravel patch in 28 m, 45 km/h." /
"Climb of 6% in 8 m, you need a lower gear." / "Top of the climb, flat road ahead: the gear has to go up." /
"Uphill 6%, 38 km/h, 56 m to go." / "12 m of side road left before the course, 40 km/h."

The perfect driver `plan(world)` works in this priority order:
1. Brake if a hazard is within 30 m and you are above 18 km/h.
2. Turn to the finish side if a fork is within 40 m.
3. Shift if the gear does not suit the terrain here or 8 m ahead.
4. Otherwise pedal.

## Tuning (perfect calls, simulated think time per call, seeds 0-2)

| Level | 0 ms (perfect) | 36 ms | 200 ms | 1.3 s |
|---|---|---|---|---|
| L0 | win 4.6-5.0 s | win 7.0-7.6 s | win 15.8-17.1 s | win 68-74 s |
| L1 | win 9.8-10.5 s | win 14.2-15.3 s | win 28.6-34.4 s | timeout, 19-23 m |
| L2 | win 9.6-10.6 s | win 13.9-15.5 s | win 30.2-36.0 s | timeout, 19-22 m |
| L3 | win 9.9-10.3 s | win 14.3-15.0 s | win 28.8-33.9 s | timeout, 21-23 m |

- A 1.3 s driver can't hold speed on the first climb (one stroke every 1.3 s loses to gravity), so it stalls.
- A 36 ms driver that only pedals and never navigates finishes L1 in about 22 s instead of 14.6 s
  (2 crashes and 2 detours), so navigation matters.
- With perfect calls the 200 ms driver never misses a fork; real mistakes come only from wrong calls.
- Replays: the perfect driver is 130-290 KB (one call and one frame per tick). Slower drivers are under 100 KB.

## ActionMan probe (laptop CPU, `s16-v3-5epb`, 12 observations per level)

| Phrasing | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| first draft | 8/12 | 3/12 | 6/12 | 2/12 |
| final | **12/12** | **12/12** | **11/12** | **11/12** |

What the first draft got wrong, and the fix:
- **Gear words in pedal lines** pulled it to `shift`, e.g. "Climbing 8% in low gear, 32 km/h." -> `shift(down)`.
  - Pedal lines no longer mention the gear.
  - The shift description no longer says "flat road" ("a low gear for climbs, a high gear everywhere else").
- **It copied the gear word as the direction:** "...you're in high gear" -> `shift(up)`, the opposite of what's needed.
  It cannot invert. Shift lines now state the need: "you need a lower gear" / "the gear has to go up".
- **Hazards got `pedal`.** Lines starting "Hazard ahead: ..." get `brake` reliably.
- **"Detour" / "wrong branch" lines got `turn`.** These are now "side road" lines.
- **L3 alias `gear` for `shift` was a magnet:** every line got `gear(...)`. The aliases are now shifter/derailleur/cog.

Still wrong in the final probe (2/48): side-road lines on L2/L3 sometimes get `brake`/`turn`.

Caveats:
- The probe sampled shift lines thinly (one up and one down per level).
- The phrasing was tuned on similar observations, so expect somewhat lower accuracy in a full race.
