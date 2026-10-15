# ActionMan 0.1: benchmark

Release candidate `s16-v3-5epb`: ModernBERT-large plus heads, about 0.4B parameters. Plan: `docs/plans/2026-10-15-actionman-benchmark-plan.md`.
Raw summaries are in `docs/results/bench/*.json`, figures in `docs/results/figures/`, and the code in `bench/`.

## Headline

- **Speed:** on one RTX 4090 at batch 1, ActionMan takes **17 ms** (p50) per request against **121 ms** for the same-data Qwen3-0.6B served by tuned vLLM, so it is **7.1x faster**.
  The gap grows with the answer: 16.8 vs 111 ms for one call, and 21 vs 331 ms for three or more (about 16x).
- **Accuracy:** ActionMan is **less accurate** than the LLMs it is compared with.
  - On the official BFCL v4 harness it scores 66.0 Non-Live, 45.1 Live and 81.2 Irrelevance.
  - Hammer2.1-0.5b, which is about the same size, scores 65.9 / 54.7 / 80.7.
  - Our Qwen3-0.6B trained on the same data scores 75.7 / 54.1 / 79.1.
- **Framing (decided by the plan's triggers):**
  - **"Fast tool caller":** the GPU ratio is at least 5x.
  - **Lead with latency and calibration, not accuracy:** ActionMan's Non-Live Python score (69.6) is below granite-4.0-350m (73.0).

## Table 1: BFCL v4, official harness (single-turn categories)

`bfcl-eval==2026.3.23`, run on one box with one harness. The leaderboard's own formulas are used:
- **Non-Live** = mean(simple(py, java, js), multiple, parallel, parallel_multiple);
- **Live** is count-weighted;
- **Irrelevance** = mean(irrelevance, live_irrelevance).

| model | params | Non-Live | Non-Live, Python only | Live | Irrelevance | Relevance |
|---|---|---|---|---|---|---|
| **ActionMan 0.1** | 0.4B | 66.02 | 69.56 | 45.08 | **81.20** | 62.50 |
| Hammer2.1-0.5b | 0.49B | 65.94 | 71.31 | 54.70 | 80.74 | 68.75 |
| granite-4.0-350m | 0.35B | 67.67 | 73.00 | 46.71 | 61.07 | 81.25 |
| Qwen3-0.6B (FC) | 0.75B | 72.85 | 77.81 | 58.77 | 80.92 | 68.75 |
| Qwen3-0.6B SFT on ActionMan's data | 0.75B | 75.67 | 80.50 | 54.11 | 79.10 | 75.00 |
| Qwen3-1.7B (FC) | 2.0B | 82.67 | 88.25 | 74.69 | 76.79 | 75.00 |
| GPT-6 Luna (API, reasoning low) | - | 78.21 | 82.12 | 68.76 | 75.60 | 68.75 |

Per category (ActionMan, with the bootstrap 95% CI):

| category | n | ActionMan | 95% CI | Qwen3-0.6B SFT | Hammer2.1-0.5b |
|---|---|---|---|---|---|
| simple_python | 400 | 76.8 | 72.5-80.8 | 85.5 | 83.2 |
| simple_java | 100 | 55.0 | 45.0-65.0 | 59.0 | 52.0 |
| simple_javascript | 50 | 56.0 | 42.0-70.0 | 54.0 | 50.0 |
| multiple | 200 | 79.5 | 73.5-85.0 | 86.5 | 81.5 |
| parallel | 200 | 61.0 | 54.0-68.0 | 76.5 | 69.5 |
| parallel_multiple | 200 | 61.0 | 54.0-68.0 | 73.5 | 51.0 |
| live_simple | 258 | 48.1 | 41.5-54.3 | 63.6 | 56.2 |
| live_multiple | 1053 | 44.9 | 41.8-47.9 | 51.6 | 54.5 |
| live_parallel | 16 | 12.5 | 0.0-31.2 | 62.5 | 62.5 |
| live_parallel_multiple | 24 | 41.7 | 20.8-62.5 | 58.3 | 41.7 |
| irrelevance | 240 | 82.1 | 77.1-86.7 | 79.6 | 85.0 |
| live_irrelevance | 884 | 80.3 | 77.6-82.9 | 78.6 | 76.5 |
| live_relevance | 16 | 62.5 | 37.5-87.5 | 75.0 | 68.8 |

**How to read it:**
- **Harness check:** the harness reproduces the public leaderboard for the two non-thinking baselines within 0.25 Non-Live points:
  - Hammer: 65.94 vs 65.98;
  - granite: 67.67 vs 67.92.
- **Qwen3-0.6B FC:** it lands a stable 1.1-1.4 points above its leaderboard 71.79 in both of our runs. It runs in thinking mode at temperature 0.001, so its runs are sampled.
- **What is not reported:**
  - **No BFCL Overall.** Multi-turn, agentic, web search and memory were not run; ActionMan is single-turn only. Those categories count as 0 and make up most of Overall.
  - **No functiongemma-270m.** It is gated for our HF token, and its terms were not accepted.
- **Where ActionMan falls behind: Live, the real user prompts.** Its errors there are mostly:
  - span length: copying too much or too little of the query, 291 of the live_multiple argument errors;
  - reasoning: values that are not in the text, 210;
  - the wrong tool or the wrong number of calls, 154.

  Live prompts are long and multi-message. ActionMan joins every message into one query (an ablation on the last user message alone gave 45.00, the same).
- **Prompt fit:** 3,631 of 3,641 prompts fit pass A; the 10 that did not are scored as given (truncated).

## Table 2: GPU latency (the headline)

One RTX 4090 box (AMD EPYC 7502, driver 570.211), batch 1, 500 fixed BFCL v4 prompts (`bench/latency_ids.json`).
- **Timed span:** end to end, from (query, OpenAI tools) to parsed calls; model load is excluded.
- **Order:** 20 warm-up prompts first, then every arm twice in ABAB order.
- **Accuracy:** every arm's accuracy is from the official checker on the same 500 prompts.

| arm | p50 ms (A / B) | p95 ms | mean ms | accuracy on the 500 | output tokens |
|---|---|---|---|---|---|
| **ActionMan, compiled (CUDA graphs), T=3** | **17.0 / 17.1** | 28.9 | 18.2 | 59.2% | - |
| ActionMan, compiled, T=1 | 14.9 / 14.9 | 24.5 | 16.6 | 58.0% | - |
| ActionMan, eager, T=3 | 44.6 / 44.7 | 83.1 | 49.7 | 59.6% | - |
| Qwen3-0.6B SFT (same data), vLLM 0.11 bf16 | 122.4 / 119.8 | 347.4 | 150.2 | 68.6% | 51 |
| same, vLLM `max_num_seqs=1` | 120.3 / 122.6 | 352.4 | 151.2 | 68.8% | 51 |
| Qwen3-0.6B public, vLLM, no thinking | 122.0 / 120.5 | 345.2 | 151.0 | 56.0% | 51 |
| Qwen3-0.6B SFT, llama.cpp Q8_0, full GPU offload | 159.4 / 166.0 | 336.0 | 178.5 | 69.0% | 51 |
| Qwen3-0.6B SFT, HF `generate` (reference, 120 prompts) | 1,243 | 4,501 | 1,638 | 69.2% | 56 |
| GPT-6 Luna API from the laptop (100 prompts, network included, reference) | 1,455 | 2,335 | 1,541 | - | - |

Median latency by the number of calls in the gold answer (the figure `figures/latency_vs_output_size.png`):

| gold calls | n | ActionMan compiled | vLLM Qwen SFT | ratio |
|---|---|---|---|---|
| 1 | 406 | 16.8 ms | 110.6 ms | 6.6x |
| 2 | 44 | 15.3 ms | 200.1 ms | 13.1x |
| 3+ | 50 | 21.0 ms | 330.6 ms | 15.7x |

- **Why it stays flat:** ActionMan writes every argument in 2-3 parallel passes, so it barely notices how long the answer is. An LLM pays for every output token.
- **Fast but less accurate:** on these 500 prompts ActionMan is 9.4 points less accurate than the SFT. The speed claim always comes with that number.
- **Prefix caching is off on both sides.**

## Table 3: CPU latency (server CPU, same cores and threads)

The b2 box CPU (AMD EPYC 9554) was pinned with `taskset` to 8 cores, with 8 threads for both engines, on 200 of the 500 prompts.

| arm | p50 ms (A / B) | p95 ms | accuracy |
|---|---|---|---|
| **ActionMan fp32, eager, T=3** | **447 / 470** | 1,044 | 59.0% |
| Qwen3-0.6B SFT, llama.cpp Q8_0 | 1,657 / 1,636 | 3,707 | 66.5% |

- **Server CPU:** ActionMan is 3.6x faster.
- **Laptop CPU:** not measured cleanly. Both attempts ran at 66-98% background load on a battery-powered i7-1255U, which gave an indicative ~5 s per call for ActionMan. CPU speed on consumer laptops is a known weak point; int8/ONNX export is the follow-up.

## Our own eval under the official checker

These are our 2,340 prompts (1,000 xLAM test, 100 xLAM no-tool, and 1,240 BFCL v3 prompts), scored with the official BFCL AST checker.

| model | all | xLAM | xLAM no-tool | BFCL simple | multiple | parallel | par-multiple | irrelevance |
|---|---|---|---|---|---|---|---|---|
| **ActionMan 0.1** | 79.0 | 84.0 | 98.0 | 77.5 | 81.5 | 61.5 | 60.5 | 80.8 |
| Qwen3-0.6B SFT, same data | 85.6 | 89.4 | 98.0 | 86.0 | 87.0 | 77.0 | 73.0 | 80.4 |
| Qwen3-0.6B (FC) | 79.6 | 77.4 | 96.0 | 86.5 | 85.5 | 67.0 | 71.0 | 82.9 |
| Hammer2.1-0.5b | 79.0 | 81.3 | 97.0 | 83.0 | 79.0 | 70.5 | 50.5 | 86.2 |
| granite-4.0-350m | 75.5 | 75.6 | 99.0 | 73.5 | 74.0 | 69.0 | 59.5 | 88.3 |
| Qwen3-1.7B (FC) | 85.0 | 80.5 | 96.0 | 91.2 | 93.5 | 88.0 | 83.0 | 80.4 |

- **Scorer leniency is small:** our scorer gives 79.7 and the official one 79.0, a gap of 0.6 points. The 79.7 headline stands, and every public number uses the official checker.
- **Possible contamination:** Hammer may have trained on xLAM-60k, so its xLAM column may be contaminated. BFCL is the primary table.

## Calibration and abstention

| | own eval | BFCL v4 (Python) |
|---|---|---|
| ECE (min of tool and argument confidence, called prompts) | 0.11 | 0.28 |
| accuracy at 100% coverage | 78.6 | 56.1 |
| accuracy when the least confident 10% abstain | 85.1 | 62.1 |
| accuracy when the least confident 20% abstain | 89.5 | 66.9 |

- **Abstaining helps:** confidence ranks well (abstaining the least confident 20% adds 11 points on both sets), but it is over-confident on BFCL.
- **The API knob:** `predict` returns a `confidence` per call, so a caller can say "not sure" below a threshold.
- **Seed noise on our eval:** about 1.5 points (v3 seeds 77.5-78.8; the release is 79.7).

## What we do not claim

- **No BFCL Overall**, no state of the art, and no "beats LLMs" on accuracy.
- **No multi-turn or agentic support.** ActionMan is single-turn, English, at most 4 calls per request.
- **No consumer-CPU speed claim.**
- **No old 90x ratio.** It came from an unoptimised HF `generate` baseline. Only the tuned-vLLM ratio (7.1x) is used.

## Incidents that changed numbers (all fixed before these tables)

- **SFT config written by transformers 5:** it had `rope_parameters` and no `rope_theta`, so vLLM 0.11 and transformers 4.57 silently used `rope_theta=10000` instead of 1e6.
  - Every early SFT accuracy number was wrong, and so were its latencies (shorter answers).
  - The fix is `bench/sft_tf4.py`. SFT BFCL was rerun (b5) and every GPU arm was re-timed on one fresh box (b6, the table above).
- **A Python set in ActionMan output:** one prediction produced a set, which crashed JSON export at `multiple_10`. Sets and tuples now become lists, and a crash on one prompt is scored as wrong instead of stopping the run.
- **Harness gate:** it was first anchored on Qwen3-0.6B FC, whose sampled thinking runs vary by more than the 1-point tolerance. It was re-anchored on the two deterministic baselines, and both pass within 0.25.

## Reproduce

```
python bench/bfcl_run.py dump-entries --out data/bench/bfcl_v4_entries.jsonl     # bfcl-eval venv
python bench/bfcl_actionman.py ...                                               # ActionMan predictions (b3)
python bench/bfcl_run.py import --preds am_flat.jsonl --model actionman-s16 --result-dir result
bash bench/b1_bfcl.sh / bench/b2_latency.sh                                      # box jobs (infra/launch.py)
python bench/bfcl_run.py score-calls --lat <lat/*.json> --out lat_scored.json
python bench/report.py --b1 ... --b2 ... --scores ... --am-flat ... --lat-scored ... --gold-calls ...
```

Versions are recorded in each latency JSON's `meta`: torch 2.14.1, transformers 5.19.0, vLLM 0.11.0, bfcl-eval 2026.3.23.

## Cost

| item | cost |
|---|---|
| benchmark GPU boxes (b1-b6, including 4 failed starts) | $1.76 |
| GPT-6 Luna BFCL (3,641 prompts) plus latency (120) | $0.28 |
| arena GPU boxes (all events, all reruns) | $1.71 |
