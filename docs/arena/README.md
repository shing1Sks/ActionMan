# The Arena

Three real-time games played only through tool calls. **The clock is the model:**
- every action costs one 10 ms game tick;
- the model's thinking time is charged in full (`arena/run.py`).

A model that thinks for 1.3 s per call lets 130 ticks of the world go by between moves.

| event | the job | what the model sees | look |
|---|---|---|---|
| [2D Minecraft](mining.md) | dig to the diamond while zombies close in | "Dig down." | pixel art, lantern light |
| [Rally](rally.md) | drive a special stage at up to 360 km/h | "Left 4 is the next corner." | inked roadbook map |
| [Cycling](cycling.md) | every `pedal()` call is one stroke; shift, turn and brake on climbs, descents and forks | "Flat road, 4 km/h, 128 m to the finish." | golden-hour flat vector |

**Levels:** four per event.

| level | what it adds |
|---|---|
| L0 | warm-up |
| L1 | the full game |
| L2 | renamed tools and parameters (`dig` -> `mine`, `direction` -> `dir`) |
| L3 | renamed values too (`left` -> `port`), while the instruction still says left |

**Every model plays the identical world:** each (level, seed) is fixed, and every model gets one instruction per turn and must answer with a call.

## Results

12 races per event (L0-L3 x seeds 0-2).

| | ActionMan-0.1 | Qwen3-0.6B, same data | GPT-6 Luna | Perfect driver |
|---|---|---|---|---|
| **2D Minecraft:** wins | **10 / 12** | 9 / 12 | 10 / 12 | 12 / 12 |
| finish time | **0.4-2.1 s** | 4.9-19.1 s | 17.3-69.6 s | 0.1-0.6 s |
| **Rally:** wins | **11 / 12** | 9 / 12 | 3 / 12 | 12 / 12 |
| finish time | **12.9-17.1 s** | 13.9-22.8 s | 17.7-24.0 s | 12.8-15.2 s |
| **Cycling:** wins | **11 / 12** | 3 / 12 | 3 / 12 | 12 / 12 |
| finish time | **6.6-34.3 s** | 37.4-42.6 s | 72.9-78.6 s | 4.6-10.6 s |
| thinking per call (median) | **13-26 ms** | 344-459 ms | 1.2-1.3 s | - |
| invalid calls | 22 of 17,838 | 1,676 of 3,387 | 0 of 1,426 | 0 |

- **2D Minecraft:** ActionMan reaches the diamond in under 2.1 s; GPT-6 Luna takes up to 70 s.
- **Rally:** on 10 of its 11 wins, ActionMan finishes within 0.25 s of the perfect driver. GPT-6 Luna reaches each corner about a second late and retires from L1 on.
- **Cycling:** calls per second is power. ActionMan pedals at about 40 strokes per second; the LLMs run out the 90 s clock from L1 on.
- **Renamed values (L3):** Qwen sends the plain value (`down` instead of `D`), so almost every call is invalid.

## Setup

- **Hardware:**
  - ActionMan and Qwen ran on the same RTX 4090 box in one session. Qwen runs HF `generate` at batch 1.
  - GPT-6 Luna ran over its API (reasoning low) from a laptop, so its time includes the network.
- **The instructions:** a scripted planner turns the true game state into one plain instruction per turn, and every model gets the same one. The phrasing was tuned with short ActionMan probes (`rally.md`, `cycling.md`).
- **A call is required:**

  | model | how the call is forced |
  |---|---|
  | GPT-6 Luna | `tool_choice="required"` |
  | ActionMan | `require=True` |
  | Qwen | its answer is prefilled with `<tool_call>` |

## Run it

```
python -m arena.run --games mining rally cycling --agents scripted actionman --levels 0 1 2 3 --seeds 0 1 2 \
    --ckpt runs/hf/runs/s16-v3-5epb/m --style order
cd arena/viewer && python -m http.server 8931 --bind 127.0.0.1      # open http://127.0.0.1:8931
```

- **Agents:** `scripted`, `random`, `actionman`, `qwen`, and `openai`, which reads `OPENAI_API_KEY` from `.env`.
- **Replays** land in `arena/replays/`, and `arena/viewer/replays.js` is rebuilt from them.
