# The films

Two films of ActionMan-0.1. Each is one animated page, rendered frame by frame to MP4 (1920x1080, 30 fps), with a soundtrack
synthesised in numpy.

| film | page | length | what it is |
|---|---|---|---|
| **The launch film** | `launch.html` | 60 s | Instruction → Run, "Decision models choose. Action models act.", masked diffusion vs autoregressive decoding layer by layer, speed against Qwen3-0.6B and GPT-6 Luna, the three Arena races, wins and accuracy |
| The research cut | `index.html` | 128 s | the longer version, with the run history |

**Everything on screen is real:**
- every number comes from `data.json`;
- the inside scene and the cold open replay `trace.json`. That is the released checkpoint run on real BFCL items, pass by pass:
  tool probabilities, copy spans, every slot token and its confidence. It also holds the tokens the Qwen3-0.6B baseline writes
  for the same call;
- the races are drawn by the Arena's own renderers (`../arena/viewer/games/*.js`) from `../arena/viewer/replays.js`.

| File | What |
|---|---|
| `launch.html`, `index.html` | the films: each one paused GSAP timeline, exposes `window.seek(t)`, `window.DURATION`, `window.ready`, `window.CUES` |
| `data.json` | every on-screen number, with a `_source` per block |
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

# the launch film: cues -> video -> music -> mux
node render.cjs --page launch.html --cues launch-cues.json
node render.cjs --page launch.html --out launch-video.mp4 --preset medium --threads 4
python make_music.py launch-cues.json launch-music.wav
ffmpeg -y -i launch-video.mp4 -i launch-music.wav -map 0:v -map 1:a -c:v copy -af loudnorm=I=-14:TP=-1 -c:a aac -b:a 192k -shortest actionman-launch-film.mp4

# the research cut (index.html is the default page)
node render.cjs --cues cues.json && node render.cjs --out film-video.mp4 && python make_music.py
ffmpeg -y -i film-video.mp4 -i music.wav -map 0:v -map 1:a -c:v copy -af loudnorm=I=-14:TP=-1 -c:a aac -b:a 192k -shortest actionman-research-film.mp4

# one section only, while iterating
node render.cjs --from 87 --to 96 --out race.mp4

# the mechanism trace (CPU, ~20 s per example after a ~5 min model load); from the repo root
PYTHONPATH=src .venv/Scripts/python.exe video/tools/trace_example.py --id multiple_137 multiple_12 multiple_187 parallel_0
```

`render.cjs` uses `C:\Program Files\Google\Chrome\Application\chrome.exe` when it exists; otherwise pass
`--channel msedge` (or `--chrome <path>`).

## Gotchas

- `window.seek` must return nothing (Playwright would serialise the GSAP timeline and crash the tab).
- Everything that changes per frame (typewriter, slot tiles, race lanes, strips) is a pure function of film time in `DYN`,
  never a CSS transition, so seeking is frame-exact.
- Pixelify Sans is only used for the big title words: at small sizes its digits misread (5 looks like 8). Numbers use
  Instrument Sans or JetBrains Mono (with ligatures off, or `<|im_end|>` turns into arrows).
- `index.html` loads `../arena/viewer/*` by relative path, so keep `video/` inside the repo. Missing renderers or replays show a
  placeholder card instead of failing.
- Fonts come from Google Fonts: the render needs network once (`window.ready` waits for them).
