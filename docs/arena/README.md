# ActionMan Arena

Three small real-time games played only through tool calls. **The clock is the model:**
- every action costs one 10 ms game tick;
- the rest of the time is the model's real thinking time, charged in full (`arena/run.py`).

So a model that thinks for 1.3 s per call lets 130 ticks of the world go by between moves.

It is a showcase, not a benchmark. The rigorous numbers are in `docs/results/benchmark.md`.

| event | the job | what the driver sees | visuals |
|---|---|---|---|
| [Mining](mining.md) | dig to a diamond while zombies close in | one order per turn: "Dig down." | pixel art, lantern light |
| [Rally](rally.md) | drive a stage at up to 360 km/h | one pace note per turn: "Left 4 corner." | inked roadbook map |
| [Cycling](cycling.md) | one call is one pedal stroke; shift, turn and brake on a 80-145 m course | one line per turn: "Climbing 8% ..." | golden-hour flat vector |

**Common to all three events:**
- **Levels:** four per event. L0 is a warm-up, and L1 is the full game.
- **Renamed controls:** L2 renames the tools and parameters, and L3 renames the values too (`left` -> `port`), while the notes still say left.
- **Same world for everyone:** each (level, seed) is a fixed world, so every driver gets the identical map.
- **A call is required:** GPT uses `tool_choice="required"`, ActionMan `require=True`, and Qwen gets its answer prefilled with `<tool_call>`.

## Results (12 races per event: L0-L3 x seeds 0-2)

ActionMan and Qwen ran on the same RTX 4090 box in one session (`arena-final`). GPT-6 Luna ran over the API from the laptop, so its time includes the network.

| | ActionMan 0.1 (0.4B) | Qwen3-0.6B SFT, same data | GPT-6 Luna (API, reasoning low) | Perfect driver |
|---|---|---|---|---|
| **Mining:** won | **10 / 12** | 9 / 12 | **10 / 12** | 12 / 12 |
| finish time (wins) | **0.4-2.1 s** | 4.9-19.1 s | 17.3-69.6 s | 0.1-0.6 s |
| **Rally:** won | **11 / 12** | 9 / 12 | 3 / 12 (L0 only) | 12 / 12 |
| finish time (wins) | 12.9-17.1 s | 13.9-22.8 s | 17.7-24.0 s | 12.8-15.2 s |
| **Cycling:** won | **11 / 12** | 3 / 12 (L0 only) | 3 / 12 (L0 only) | 12 / 12 |
| finish time (wins) | 6.6-34.3 s | 37.4-42.6 s | 72.9-78.6 s | 4.6-10.6 s |
| thinking per call (p50) | 13-26 ms | 344-459 ms | 1.2-1.3 s | - |
| invalid calls | 22 of 17,838 | 1,676 of 3,387 | 0 of 1,426 | 0 |

**What the table shows:**
- **Speed decides the game when the world moves.**
  - GPT-6 Luna is the most reliable caller here: it never sent an invalid call, even with renamed controls.
  - Mining forgives slow thinking, and GPT wins 10 of 12.
  - In rally and cycling it arrives at each corner or fork about a second late and retires or runs out of time on every level past the warm-up.
- **ActionMan is fast enough to keep up**, and close to the perfect driver's times in rally.
- **Qwen loses in two ways:**
  - On L3 it sends the plain value (`down` instead of `D`), so almost every call is invalid.
  - In cycling, where calls per second is pedal power, its 0.3-0.6 s per stroke runs out the 90 s clock from L1 on.
- **ActionMan's losses:**
  - two mining races to zombies, while it looped on a wrong value for "below you" on L3;
  - one rally crash-out on L3;
  - one cycling timeout on L2.

## How fair is it

- **The calls are translations, not plans.** ActionMan does not plan, so a scripted planner turns the true state into a plain instruction each turn, and every driver gets the same instruction.
  - It was measured (`mining.md`): from a description of the state, ActionMan wins 1 of 12 mining races.
  - This tests the thing ActionMan is built for: turning a request into a correct call, fast.
- **The instructions were tuned on ActionMan.** The phrasing was tuned with short ActionMan probes (`rally.md`, `cycling.md`), and the other drivers did not get the same tuning.
- **Hardware:**
  - ActionMan and Qwen share one GPU box. Qwen runs HF `generate` at batch 1. Served by tuned vLLM it would be several times faster, and still about 7x slower than ActionMan (`benchmark.md`).
  - GPT-6 Luna's time includes the network from the laptop.

## Run it

```
python -m arena.run --games mining rally cycling --agents scripted actionman --levels 0 1 2 3 --seeds 0 1 2 \
    --ckpt runs/hf/runs/s16-v3-5epb/m --style order
cd arena/viewer && python -m http.server 8931 --bind 127.0.0.1      # open http://127.0.0.1:8931
```

Agents: `scripted`, `random`, `actionman`, `qwen`, and `openai` (which reads `OPENAI_API_KEY` from `.env`). Replays land in `arena/replays/`; `arena/viewer/replays.js` is rebuilt from them.

Cost of every arena run, including the reruns: $1.71 of GPU time.
