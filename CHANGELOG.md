# Changelog / decision log

## 0.1 (2026-10-15): first release, ActionMan-0.1

- **Model:** ActionMan-0.1, an action model.
  - ModernBERT-large encoder (28 layers, 395M weights) with a decision head over up to 4 `[CALL]` markers and a length head.
  - Arguments are written by masked diffusion over a slot block, with copy, option and list slots.
  - Trained for 5 epochs on 80,459 rows: xLAM-60k, ToolACE, Hermes v1 and 11k no-call examples.
- **Why masked diffusion, not autoregressive decoding:** every argument is denoised in parallel in 2-3 forward passes, so latency
  is flat in the size of the answer.
  - One call takes 17 ms, vs 122 ms for a same-data Qwen3-0.6B on tuned vLLM (RTX 4090, batch 1).
  - Three or more calls take 21 ms vs 331 ms (`docs/results/benchmark.md`).
- **Why copy and option slots:** 80-92% of BFCL string values appear word for word in the instruction.
  - Copying spans removes misspelled values (garbled values went from 77 to 13).
  - Enums and booleans are picked from their options, not spelled out.
- **Why 5 epochs:** the best of every recipe we measured on our 2,340-prompt eval (79.7, against seed noise of about 1.5 points;
  `docs/results/findings.md`).
- **Why one model, with no LLM fallback:** a fallback breaks the size and speed promise. Each call returns a `confidence` that can
  be thresholded instead.
- **Benchmarks:** official BFCL v4 harness (`bfcl-eval==2026.3.23`), single-turn categories, validated against the public
  leaderboard on Hammer and granite (within 0.25 points).
- **The Arena:** three real-time games (2D Minecraft, rally, cycling) where thinking time is charged to the model.
- **Licence:**
  - The code is MIT.
  - The weights are for research use only (xLAM-60k).
  - A retrain without xLAM would allow open weights.
