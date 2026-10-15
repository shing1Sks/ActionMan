# Changelog / decision log

## 0.1 (2026-10-15): first research preview

- **Model:** ActionMan-0.1 (`s16-v3-5epb`).
  - ModernBERT-large (395M) with a decision head (up to 4 `[CALL]` markers), a length head, and a masked-diffusion slot block.
  - The slot block has a copy switch, option slots and list slots.
  - Trained for 5 epochs on train_v2: 80,459 rows of xLAM-60k, ToolACE, Hermes v1 and 11k no-call negatives.
- **Why one encoder plus a slot block, not an LLM:** the latency is flat in the length of the answer.
  - 17 ms vs 121 ms for tuned vLLM at batch 1 on an RTX 4090.
  - For three or more calls it is 21 vs 331 ms (`docs/results/benchmark.md`).
- **Why copy and pick (v3):** 80-92% of BFCL string values occur in the query word for word.
  - Copying cut garbled values from 77 to 13.
  - It moved the eval from 75.9 to 78.6 (findings, section 1).
- **Why 5 epochs:** it was the only change outside the ~1.5-point seed noise (79.7) (findings, section 0).
- **Why no LLM hand-off:**
  - Escalating the least confident 17% to Qwen reached 84.0%, but it breaks the one-small-model promise. It is kept as a measured result only.
  - The released `confidence` lets a caller abstain instead.
- **Dropped:**
  - distillation from Qwen3-14B (-1.0, inside noise);
  - train_v4 with +80% data (78.7);
  - word-boundary snapping (-0.1 to -0.2 on all 7 checkpoints);
  - v3.1 masking (inside noise; the code path is kept).
- **Benchmarks:** official BFCL v4 harness (`bfcl-eval==2026.3.23`), validated against the public leaderboard on Hammer and granite (within 0.25).
  - No Overall score: multi-turn and agentic categories were not run.
  - functiongemma-270m was not evaluated (gated).
- **Arena:** three real-time events (mining, rally, cycling) where thinking time is charged to the model. It is a showcase, not a benchmark.
- **Not released:** the prepared training data (the xLAM licence) and the private run repo with checkpoints and logs.
- **Licence:**
  - The code is MIT.
  - The weights are a research preview for research use only, because train_v2 includes xLAM-60k (research only).
  - A clean retrain without xLAM is the path to open weights.
