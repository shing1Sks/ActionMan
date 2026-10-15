# Arena event: Rally

Drive a point-to-point special stage from the co-driver's pace notes. Files: `arena/rally/` (world, observer, plan),
`arena/viewer/games/rally.js`, `tests/test_arena_rally.py`.

## Rules
- The stage is generated per (level, seed): straights and corners of severity 1-6 (1 = hairpin, 6 = kink).
  Headings are steered back toward east so it runs point to point, and it is redrawn until no two stretches come within 26 m.
- The car never stops: every 10 ms tick it covers v/100 m. Tools: `steer(direction, severity)` sets the line for the
  next corner, `brake()` commits to braking for a hairpin, `throttle()` asks for top speed. Every action costs one tick.
- At each corner entry the line must match (same direction, severity within 1). Hairpins must be entered at 25 m/s or
  less. A miss is a crash: the car stops for 1.5 s, restarts at 15 m/s, and needs a new `throttle()`. The line clears
  at every entry, so every corner needs its own call. 5 crashes means retired (`dead`), 120 s means `timeout`, and the finish line is `win`.
- `brake()` before a hairpin holds speed until the last braking point, so braking early costs nothing and braking late still crashes.

## Levels
| | top speed | stage | corners |
|---|---|---|---|
| L0 | 70 m/s | ~0.9 km | 3 singles (gaps 90-130 m), 1 hairpin + exit corner |
| L1 | 100 m/s | ~1.3 km | 4 singles (gaps 40-95 m), 2 complexes (2-3 corners 4-16 m apart), 1 hairpin, 1 long straight |
| L2 | as L1 | | tools and parameters renamed |
| L3 | as L1 | | left/right renamed too (port/starboard or L/R); the notes still say left/right |

## Observation
One pace note per turn, from `plan()`, with digits:
- the next corner not yet called: `Next corner: left 4.`, `Caution, right 2 corner.`, `Hairpin left 1.`
- a hairpin inside the braking point plus a 35 m margin: `Brake! Hairpin in 80.`
- back up to speed after a hairpin or crash: `Long straight, 300, flat out.`, `Full throttle, 100 clear.`
- otherwise the called corner again (the perfect driver holds its line), or `Flat out to the finish!`

Corner notes carry one number, the severity, and say "corner". Finish notes carry no number. The live distance lives on
the viewer's note card instead (see the probe below for why).

## Tuning (fake drivers: the perfect call after a fixed think time, ±15%, seeds 0-2)
| driver | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| perfect (0 ms) | win 12.8-13.7 s | win 13.9-15.2 s | win 14.0-14.9 s | win 13.5-15.0 s |
| 36 ms | win, 0 crashes, +0.1 s | win, 0 crashes | win, 0 crashes | win, 0 crashes |
| 200 ms | win, 0 crashes | win, 1-2 crashes, +3-5 s | win, 1 crash | win, 1-2 crashes |
| 1.3 s | win, 1-2 crashes, 17-21 s | retired 3/3 | retired 2/3 (one win with 4 crashes) | retired 3/3 |

Gaps inside complexes are 0.25-0.45 s at 360 km/h, which is where 200 ms runs late. Other gaps are 0.7-1.4 s, which 1.3 s misses.
Replays: perfect driver 290-345 KB (it calls every tick), 36 ms about 100 KB.

## Viewer
An inked roadbook on survey paper: fbm contours, forest stipple, spot heights, a 200 m grid, the road as inked
edges with a pale surface, hatching outside tight corners, numbered roundels (red for 1-2), and km posts at 0.25 km.
The camera frames the car and the next corner, smoothed over about 1 s of replay. The racing line is drawn behind the car:
outside, apex, outside. A late call (made less than 22 m before the corner) runs wide and wobbles. A crash slides off the outside with
skid marks, dust and a red-ink cross. The note card shows a roadbook tulip of the next corner, its live distance, the note the
driver is answering, and an amber think bar. A ring fills around the car while it thinks. An inset shows the whole stage.

## ActionMan probe (laptop CPU, 12 notes per level, exact canonical call)
Notes were sampled from perfect-driver runs on seeds 0-7, with one level's tools each: 6 corners, 2 hairpins, 2 brakes,
1 throttle and 1 finish. A hit means the call translates to the exact planned action, with the raw severity equal to the note's.

| phrasing | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| v1: `Left 4 in 60.`, brake tool mentions hairpins | 8/12 (stopped early) | | | |
| v2: corner notes without distance | 10/12 | 10/12 | 11/12 | 12/12 |
| v3 (shipped): corner notes say "corner", finish notes have no number | **12/12** | **12/12** | **10/12** | **12/12** |

Right: it maps left/right to `L`/`R` on L3 (seed 0 draws L/R; port/starboard was not probed), copies severity digits, and brakes or throttles on cue.
Wrong, and fixed by phrasing:
- v1: with two numbers in a note it copied the distance (`Right 2, 70 metres.` gave `severity=70`).
- v1: `Left 1 hairpin` gave `brake()` while the brake description mentioned hairpins.
- v2: `Flat out to the finish, 40.` gave `steer(severity=40)`; bare `Left 5.` sometimes gave `brake()`.

Still wrong: L2 seed 0 renames severity to `sharpness`, and ActionMan answers `sharpness=3` whatever the note says
(right 2 passes within 1, left 5 crashes).
