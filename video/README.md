# The research film

A ~128 s research film of ActionMan-0.1 as one animated page, rendered frame by frame to MP4 (1920x1080, 30 fps) with a
soundtrack synthesised in numpy. Same pipeline as the Piccaso film; its own look (the Arena palette and fonts).

Everything on screen is real or marked as pending:
- every number comes from `data.json` (null shows as `--`); the BFCL v4 and vLLM latency numbers are still `null`;
- the mechanism scene and the cold open replay `trace.json`: the release-candidate checkpoint (`s16-v3-5epb`) run on CPU on
  real BFCL items, pass by pass (tool probabilities, copy spans, every slot token and its confidence), plus the exact tokens
  the Qwen3-0.6B baseline writes for the same call;
- the races are drawn by the arena's own renderers (`../arena/viewer/games/*.js`) from `../arena/viewer/replays.js`, so
  they update when the replays are regenerated. Finishing times and ms per call are read from the replays.

| File | What |
|---|---|
| `index.html` | the film: one paused GSAP timeline, 14 scenes, exposes `window.seek(t)`, `window.DURATION`, `window.ready`, `window.CUES` |
| `data.json` | every on-screen number, with a `_source` per block and `"_status": "draft"` |
| `build_data_js.py` | inlines `data.json` + `trace.json` into `data.js` so the page works from `file://` |
| `tools/trace_example.py` | runs the real model on CPU and writes `trace.json` (`--scan N` lists candidate examples) |
| `render.cjs` | seeks the timeline frame by frame, pipes screenshots into ffmpeg; also writes stills and the sound cue list |
| `make_music.py` | the soundtrack (120 bpm, A minor, pulse-wave arps), numpy only; effects placed from `cues.json` |
| `gsap.min.js` | GSAP 3 (copied from the Piccaso film) |

## Commands (from `video/`)

```bash
# once
npm i playwright-core                      # already in node_modules/ here; ffmpeg must be on PATH

# after editing data.json (or re-running the trace)
python build_data_js.py                    # -> data.js
start index.html                           # plays in the browser; index.html#t=87 starts at 87 s

# stills to check a scene (PNG, into stills/)
node render.cjs --stills 8.5,33,53,70,79,85.5,94.5,126

# full render: cues -> video -> music -> mux
node render.cjs --cues cues.json           # sound-effect times exported by the page itself
node render.cjs --out film-video.mp4       # ~3,840 frames
python make_music.py                       # -> music.wav (reads cues.json)
ffmpeg -y -i film-video.mp4 -i music.wav -map 0:v -map 1:a -c:v copy -af loudnorm=I=-14:TP=-1 -c:a aac -b:a 192k -shortest actionman-research-film-draft.mp4

# one section only, while iterating
node render.cjs --from 87 --to 96 --out race.mp4

# the mechanism trace (CPU, ~20 s per example after a ~5 min model load); from the repo root
PYTHONPATH=src .venv/Scripts/python.exe video/tools/trace_example.py --id multiple_137 multiple_12 multiple_187 parallel_0
```

`render.cjs` uses `C:\Program Files\Google\Chrome\Application\chrome.exe` when it exists; otherwise pass
`--channel msedge` (or `--chrome <path>`).

## Filling in the final numbers

1. Copy the BFCL v4 harness scores into `bfcl.rows[].harness` and the latency p50s into `latency.rows[]`
   (from `docs/results/benchmark.md` / `docs/results/bench/*.json`). Set `bfcl._status` / `latency._status` to `"final"` to drop the
   "running tonight" tag, fill `latency.cpu.llama_cpp_qwen_s`, and `gap.benchmark_cost_usd` if the cost line should include it.
2. Set `end.github` / `end.huggingface`; when everything is final set `"_status": "final"` (removes the DRAFT badge).
3. Regenerate the arena replays (`arena/viewer/replays.js`); pick races in `arena.picks` (`{game, level, seed}`).
   Lanes are drawn for the agents in `arena.agents` that have a replay; a missing one is named on screen as pending.
4. `python build_data_js.py`, then the full render commands above. The music re-times itself from the new `cues.json`.

## Gotchas

- `window.seek` must return nothing (Playwright would serialise the GSAP timeline and crash the tab).
- Everything that changes per frame (typewriter, slot tiles, race lanes, strips) is a pure function of film time in `DYN`,
  never a CSS transition, so seeking is frame-exact.
- Pixelify Sans is only used for the big title words: at small sizes its digits misread (5 looks like 8). Numbers use
  Instrument Sans or JetBrains Mono (with ligatures off, or `<|im_end|>` turns into arrows).
- `index.html` loads `../arena/viewer/*` by relative path, so keep `video/` inside the repo. Missing renderers or replays show a
  placeholder card instead of failing.
- Fonts come from Google Fonts: the render needs network once (`window.ready` waits for them).
