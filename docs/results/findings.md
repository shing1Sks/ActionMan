# ActionMan: findings

- v4 experiments (2026-10-15): seed noise, copy-boundary snap, 5 epochs, more data. Section 0.
- v3 (2026-10-14 night): copy switch, option slots, list slots, distillation, escalation, compiled inference. Section 1.
- v1/v2 (2026-10-13 night): the original architecture and the Qwen baseline. Section 2.

# 0. v4 experiments (2026-10-15, plan: `docs/plans/2026-10-15-actionman-v4-plan.md`)

Same 2,340 eval prompts and scorer. Every number is full-call accuracy at T=3 on one RTX 4090 run.

| run | change | all | xLAM | BFCL simple | BFCL multiple | BFCL parallel | BFCL par-multiple | BFCL irrelevance |
|---|---|---|---|---|---|---|---|---|
| v3 seed 0 (s10) | reference | 78.6 | | | | | | 84.6 |
| v3 seed 1 | seed | 78.8 | 83.3 | 78.5 | 78.0 | 58.5 | 63.5 | 82.9 |
| v3 seed 2 | seed | 77.5 | 81.8 | 77.0 | 79.0 | 58.0 | 57.5 | 82.9 |
| v3.1 seed 0 (s12) | clean masking | 77.6 | | | | | | |
| v3.1 seed 1 | seed | 79.1 | 84.2 | 78.5 | 76.5 | 62.5 | 59.0 | 83.3 |
| v3.1 seed 2 | seed | 78.9 | 83.1 | 77.5 | 81.0 | 62.5 | 61.0 | 82.9 |
| **v3, 5 epochs** | train_v2 x5 | **79.7** | 85.2 | 78.0 | 81.5 | 61.5 | 61.0 | 80.8 |
| v3, train_v4 | 145k rows x3 | 78.7 | 83.3 | 76.5 | 80.5 | 62.5 | 59.0 | 83.8 |

- **Seed noise is about 1.5 pts.** v3 is 78.3 on average (77.5-78.8 over seeds), v3.1 is 78.5 (77.6-79.1). The v3 vs v3.1 gap from 10-14 was noise. The BFCL sub-sets (200 prompts each) swing up to 6 pts between seeds; do not read single-category moves.
- **5 epochs is the only change outside the noise band:** 79.7, +1.4 over the v3 mean and 0.9 above its best seed. That is one run; it gives up 2 pts of irrelevance (80.8).
- **More data (train_v4) did not help:** 78.7 sits inside the v3 range. The added Synth-APIGen / Glaive / Toucan rows do not transfer to this eval at 3 epochs.
- **Snap (whole-word copy widening) is dropped.** It was 0.1-0.2 pts worse on all 7 checkpoints, with no fix outweighing the breaks.
- **Cost:** $5.83 of credit ($23.00 -> $17.17), of which ~$0.76 went to three boxes stuck in Vast "loading".

# 1. v3 (2026-10-14, plan: `docs/plans/2026-10-14-actionman-v3-plan.md`)

## Headline

Same 2,340 eval prompts and scorer as v2. Qwen3-0.6B was trained on the same v2 data. All latency numbers come from one RTX 4090 box at batch 1. That host was ~2x slower in absolute terms than the v2 benchmark box, so only compare ratios across the two nights.

| | ActionMan v2 | **ActionMan v3** | v3 + escalation (theta 0.7) | Qwen3-0.6B SFT |
|---|---|---|---|---|
| Full-call accuracy (all) | 75.9% | **78.6%** | **84.0%** (17% of prompts go to Qwen) | 86.5% |
| Tool-name accuracy | 95.0% | 94.8% | - | 95.5% |
| BFCL irrelevance | 83.3% | 84.6% | 85.0% | 80.0% |
| Latency p50 / mean, eager | 89 / 86 ms | 68 / 79 ms | 63 / 375 ms | 1,633 / 1,665 ms |
| Latency p50 / mean, compiled (CUDA graphs) | 15.5 / 15.9 ms | **16.7 / 18.4 ms** | - | (not compiled) |

- **v3 closes about a quarter of the gap to the LLM:** 75.9 -> 78.6 (+2.7). It does this by picking instead of spelling.
- **Compiled inference is ~4x faster than eager at the same accuracy:** 78.5% compiled vs 78.6% eager. It runs about 90x faster than the uncompiled Qwen baseline on the same box (p50 16.7 vs 1,633 ms). Qwen was not compiled; an optimised LLM server would narrow this ratio.
- **Escalating the least confident 17% to Qwen gives 84.0%,** within 2.5 pts of Qwen. The median call stays at 63 ms (eager) because 83% of calls never touch the LLM.

## What v3 changed (and the data audit that justified it)

Audit of gold argument values (`docs/results/audit.md`, laptop, $0):
- On BFCL, 80-92% of string values and 69-86% of numeric values occur word-for-word in the query, so they can be copied.
- 5-8% of BFCL args are bools or declared enums.
- 6-8% of BFCL args are lists.

All three gates passed, so all three parts were built:

1. **Copy switch.**
   - Every value's first slot (the anchor) predicts copy vs generate.
   - On copy, a span pointer picks start/end query tokens (Jev-style dot products against the query's token states).
   - The text is sliced from the original query, so it can't be misspelled.
   - Generated values still use masked-diffusion refinement.
2. **Option slots.** Bools and enums (<= 16 options) are picked Jev-style against `[OPT]` markers written into the slot block.
3. **List slots.**
   - A count head (pass A) predicts 1..7 items, and each item gets 8 slots plus its own copy switch.
   - Lists with more than 7 items, non-scalar items, or long non-copyable items use the v2 text path.

| Model (T=3) | All | xLAM test | BFCL simple | BFCL multiple | BFCL parallel | BFCL par-multi | BFCL irrelevance | Tool acc | avg passes |
|---|---|---|---|---|---|---|---|---|---|
| v2 | 75.9 | 81.8 | 70.5 | 71.5 | 61.5 | 56.0 | 83.3 | 95.0 | 2.45 |
| **v3** | **78.6** | **83.7** | **76.2** | **80.5** | 58.5 | **59.5** | **84.6** | 94.8 | **2.17** |
| v3, compiled | 78.5 | 83.7 | 76.0 | 80.0 | 59.5 | 58.0 | 84.6 | 94.8 | 2.17 |
| v3 on distilled data (v3d) | 77.6 | 82.8 | 77.0 | 78.0 | 61.0 | 56.0 | 80.4 | 94.5 | 2.18 |
| v3.1 (masking fix from final review) | 77.6 | 82.5 | 79.2 | 80.5 | 59.0 | 56.5 | 76.7 | 93.6 | 2.20 |

- **The biggest gains are where copying matters:** BFCL simple +5.7, multiple +9.0. Parallel -3.0 is within the ~3.5 pt noise at n=200.
- **A single pass is now nearly enough:** T1 76.9% (v2 T1 71.0%). Copied and picked values need no refinement, so v3 also averages fewer passes (2.17 vs 2.45).

**Error classes** (`src/actionman/errclass.py`): every wrong argument in a call whose tool was right, same 2,340 prompts.

| class | v2 | v3 | |
|---|---|---|---|
| garbled (answer was in the query, output misspelled it: "Teslala", "Lion Messi") | 77 | **13** | -83% |
| list format (bracket/comma soup) | 74 | **35** | -53% |
| truncated ("LeBron-") | 61 | **41** | -33% |
| span length (copied a real phrase, wrong boundary: "90 minutes" for 90, "#funnyvideos") | 68 | 85 | +25% (new failure mode of copying) |
| reasoning (answer not in the query: 11 PM -> 23, unit conversion, knowledge) | 175 | 173 | unchanged, as expected |
| wrong tool (whole call) | 118 | 122 | unchanged |

Garbling is essentially solved. Most of what remains is reasoning and conversion, which an encoder this size can't do. That is the case for escalation.

**v3.1: the final-review masking fix.**
- **What it fixes:**
  - v3's training never revealed a value's anchor slot (inference does, in refinement);
  - v3 trained the copy gate on partly revealed values (inference reads it with the value fully masked).
- **How:** v3.1 fully masks a random half of the values, trains the gate/span only on those, and lets the rest follow the normal random mask.
- **Result:** 77.6% vs 78.6%, a mixed picture.

| error class | v3 | v3.1 |
|---|---|---|
| garbled | 13 | 8 |
| span length | 85 | 78 |
| list format | 35 | 26 |
| wrong tool | 122 | **149** |

- The argument side improved, and BFCL simple rose +3.0.
- The loss is all in tool choice and irrelevance (84.6 -> 76.7). The fix does not touch those directly, although in the cached variant every loss shares one forward.
- With single runs I can't separate this from seed variance. v3 stays the reference. The fix stays in the code because it makes training match inference.
- **Next:** multi-seed runs of v3 vs v3.1.

## Distillation (teacher relabel)

- **Teacher:** Qwen3-14B-FP8 under vLLM on an RTX 5090.
  - The 32B teacher was not usable tonight: the only H100 listing was stale and the H100 SXMs had CUDA 12.2/12.4.
  - It relabelled all 68,094 training examples that contain calls in 58 minutes (34.6M prompt tokens).
- **Merge rule:** keep the teacher's surface form only where it states the same fact as the original label; otherwise keep the original.
- **Result:**
  - 95.4% of calls were accepted;
  - **2.93% of argument values changed**;
  - 2,810 query-copyable args were added.
- **What changed:** mostly casing ("atp" -> "ATP"), plus some format ("London,uk" -> "London, UK", "x**2" -> "x^2").
- **The teacher is not clearly more consistent than the original labels.** "Berlin, Germany" -> "Berlin" but "Paris" -> "Paris,France". The scorer already ignores case and punctuation, so I expected a small effect.

**Result: distillation did not help.**
- v3d (v3 recipe, distilled data) scores **77.6%** vs v3's 78.6% (-1.0, within noise).
- The clearest move is BFCL irrelevance at -4.2 (80.4 vs 84.6).
- Reasoning-class errors went up (199 vs 173).
- The S7 gate (needed +2 pts) was not met, so Qwen was not retrained on the distilled data.

**Caveat found in final review:** the merge rule's string-containment test was too loose.
- It accepted fact changes such as page "1" -> "10" and "AAPL" -> "AAPL,MSFT".
- Re-merging with the fixed rule keeps 5,207 of the 8,305 teacher edits, so ~3,100 suspect edits (~1.6% of all argument values) were in v3d's training data. That may explain part of the drop.
- A strictly merged file is ready (`data/prepared/train_v2_distilled_strict.jsonl`) but untrained.

Either way, the surface-form relabel touches a class of errors (case/format) that the scorer already forgives. Distillation in this form is not the lever.

## Escalation: the near-LLM-accuracy dial

ActionMan v3 answers every prompt. A prompt is "shaky" if any call has min(tool conf, arg confs) < theta or an argument overflowed. Shaky prompts go to the Qwen3-0.6B baseline (same v2 data).
- **full:** Qwen redoes the prompt from scratch.
- **assisted:** Qwen sees only ActionMan's chosen tools, and its answer is pre-filled with every confident call plus the shaky call up to its first shaky value.

| theta | escalated | full: accuracy | assisted: accuracy | full: mean latency | assisted: mean latency |
|---|---|---|---|---|---|
| (none) | 0% | 78.6 | 78.6 | 70 ms | 70 ms |
| 0.3 | 4.6% | 80.2 | 79.1 | - | - |
| 0.5 | 9.7% | 82.4 | 79.1 | 234 ms | 114 ms |
| 0.7 | 17.1% | **84.0** | 78.1 | 375 ms | 148 ms |
| 0.85 | 23.3% | **85.0** | 77.7 | - | - |
| Qwen alone | 100% | 86.5 | - | 1,665 ms | - |

- **full works.** At theta 0.7 it gets 84.0% (Qwen 86.5%) at a mean latency 4.4x lower than Qwen's, with the median unchanged.
- **assisted does not.** Pre-filling ActionMan's confident values and cutting the tool list gives no gain. I think two things are going on:
  - the "confident" values it pins in are sometimes wrong;
  - Qwen was never trained on partial prefixes or tool subsets.

  Reusing ActionMan's answers would need Qwen fine-tuned on that format; that's an open question, not a result.
- Escalation latency was measured eager on both models. With ActionMan compiled, the no-escalation path is ~17 ms.

## Compiled inference (CUDA graphs)

- **How:** padded, static-shape versions of the cached context and slot passes (`cached.make_static`). The context is bucketed to 256 tokens and the slot block to 128, with padded keys masked. They run under `torch.compile(mode="reduce-overhead")`.
- **Correctness:** equivalent to eager on CPU (test `test_static.py`).

| T=3, same 4090 | eager p50 / mean | compiled p50 / mean | outputs identical to eager | full-call accuracy |
|---|---|---|---|---|
| v2 | 89.3 / 85.8 ms | 15.5 / 15.9 ms | 95.3% | (not re-evaluated) |
| v3 | 67.6 / 79.0 ms | 16.7 / 18.4 ms | 95.7% | 78.5% (eager 78.6%) |

- **The ~4.7% of outputs that differ are bf16 numeric noise flipping near-ties.** Full-eval accuracy is unchanged within 0.1 pt.
- **Compile cost:** a one-off warm-up of every length bucket, a few minutes, not included above.
- **Speed vs Qwen:** compiled v3 at 16.7 ms p50 is ~98x faster than eager Qwen's 1,633 ms on the same box. It was 20x last night, uncompiled on both sides.

## v3 cost and incidents

**Spend tonight: $4.50** by account credit ($27.53 -> $23.03), against a plan of ~$8 expected / $15 cap. Per-run detail is in `docs/results/costs.md` (tonight = the `s8-` .. `s14-` rows).
- That table's "no log" rows are estimates at <=30 min and overstate actual spend.
- About $1.9 of tonight's spend went to failed distillation launches and stalled hosts.

Seven distillation launches were needed. All failures were infrastructure:
- a stale H100 listing;
- pip vs a Debian-owned PyJWT (fix: vLLM in a uv venv);
- my laptop watcher killing a box in the gap before its ledger line was written (fix: 10-min grace);
- DeepGEMM and FlashInfer JIT needing `nvcc` on a runtime image (fix: `VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`);
- a host stalled on the data bundle (fix: `timeout 900`, machine marked bad);
- a host with failing DNS. Its self-destroy call also failed, so it idled ~15 min until I destroyed it (fix: self-destroy now retries 10x).

## Next (recommendation)

1. **Ship-shaped:** v3 + compiled inference + full escalation at theta 0.5-0.7. That gives ~82-84% accuracy at a median of ~17 ms, with 10-17% of calls paying the LLM cost.
2. **Span-boundary errors (85) are the largest error class v3 can still fix itself.** Two ideas:
   - a type-aware copy rule (numeric params must copy a number);
   - span-end training on whole words.

   Neither is tested.
3. **Reasoning errors (173) need a model that can reason.** Either escalate them, or fine-tune Qwen on the assisted prefix format so escalation can reuse ActionMan's answers.
4. **Distillation as tried (surface-form relabel) is not the lever:** -1.0 pt. If revisited, aim the teacher at the reasoning class (conversions, normalised dates/units) and use the strict merge rule.

# 2. ActionMan v1/v2: findings (2026-10-13, overnight away session)

Everything below was measured, not estimated. Full tables: `docs/results/results.md` (generated by
`infra/results.py`); per-run artifacts and checkpoints: private HF repo `shing-dev/actionman`, `runs/<run>/`.

## Headline

Fair comparison: both models trained on the **same v2 data** (xLAM + ToolACE + Hermes + negatives), same scorer,
same 2,340 eval prompts. Latency: same RTX 4090, batch 1, plain PyTorch / HF (no CUDA graphs, no vLLM).

| | ActionMan v2 (T=3) | Qwen3-0.6B SFT (v2 data) |
|---|---|---|
| Latency p50 / mean / p95 | **38 / 36 / 52 ms** | 777 / 1,128 / 2,871 ms |
| Sequential forward passes | 2.8 avg (early exit) | ~59 decode steps + prefill |
| Full-call accuracy (all) | 75.9% | **86.5%** |
| Tool-name accuracy | 95.0% | 95.5% |
| BFCL irrelevance (must NOT call) | **83.3%** | 80.0% |
| xLAM held-out test | 81.8% | 91.3% |

**~20x faster at p50 (~31x mean) at 88% of the LLM's full-call accuracy.** Tool choice is on par and
"don't call" is slightly better; the ~10.6-pt gap is entirely argument *values* (BFCL simple/multiple -16,
parallel -15, parallel-multiple -17). That gap is the next thing to attack (distillation; see Next).

Qwen latency was measured with the xLAM-data checkpoint (identical architecture, same ~59-token outputs); the v2
checkpoint finished after the benchmark box. The speed ratio was the same on a second, slower 4090 box (92 ms vs 1,891 ms p50): at this size both models
are bound by Python / kernel-launch overhead, so absolute ms vary by host CPU; the ratio comes from 2-3 forward
passes vs ~59 sequential decode steps.

## All models (full-call % / tool-name %, T=3 for ActionMan)

| Model | All | xLAM test | BFCL simple | BFCL multiple | BFCL parallel | BFCL par-multi | BFCL irrelevance | Tool acc |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B SFT, xLAM data | 81.8 | 91.5 | 83.8 | 83.5 | 74.5 | 67.0 | 48.8 | 91.5 |
| Qwen3-0.6B SFT, v2 data (fair) | 86.5 | 91.3 | 87.0 | 87.5 | 76.5 | 73.0 | 80.0 | 95.5 |
| ActionMan v1, standard mask | 71.2 | 82.5 | 67.7 | 67.5 | 53.5 | 48.5 | 55.0 | 92.9 |
| ActionMan v1, cached mask | 69.9 | 80.9 | 67.2 | 68.0 | 55.5 | 46.0 | 50.4 | 91.6 |
| + ToolACE/Hermes data | 75.3 | 82.2 | 73.3 | 74.0 | 58.5 | 49.0 | 77.9 | 95.0 |
| **v2: + call markers + hard negatives** | **75.9** | 81.8 | 70.5 | 71.5 | **61.5** | **56.0** | **83.3** | 95.0 |
| ablation: + data + markers only (no hard negatives) | 75.3 | 82.5 | 71.0 | 75.0 | 62.0 | 49.5 | 75.8 | 94.6 |

## Refinement passes and threshold (v2)

| | T=1 | T=2 | T=3 | T=4 | T=5 | T=3, tau=0.95 |
|---|---|---|---|---|---|---|
| Full-call accuracy (all) | 71.0 | 75.4 | 75.9 | 76.0 | 76.0 | 75.2 |
| Avg forward passes used | 1.87 | 2.22 | 2.45 | 2.64 | 2.83 | 2.62 |
| Latency p50 (ms) | 26 | 38 | 38 | — | — | — |

Masked-diffusion refinement is worth +4.9 pts and plateaus at T=3; early exit keeps the average under 2.5 passes.

## Latency: standard vs cached mask (same box)

| T=3 | p50 | mean |
|---|---|---|
| standard mask (re-encodes context every pass) | 40.0 ms | 38.9 ms |
| cached block-causal mask (slots only) | 38.2 ms | 37.3 ms |

Honest result: in eager PyTorch the cache saves only ~5%, because a pass costs ~12 ms of launch overhead
regardless of token count. Its measured wins are elsewhere: one forward per training step instead of two
(~2x cheaper training) at -1.3 pts accuracy. With CUDA graphs / fused kernels the FLOP savings should show.

## What we learned

1. **The architecture works.** A 395M encoder picks tools better than a 0.6B LLM (95.0 vs 91.5) and fills
   arguments by masked diffusion; xLAM-style exact-call accuracy 82%.
2. **Refinement passes are the diffusion payoff:** +4.9 pts from 1 to 3 passes, nearly free with early exit.
3. **Error analysis drove v2:** repeated calls to the same tool were copying each other's values (slot blocks
   indistinguishable) -> `#1/#2` markers: BFCL parallel +3.0, parallel-multiple +7.0.
4. **"Don't call" is a data problem:** ToolACE no-call examples + hard negatives took BFCL irrelevance 50 -> 83
   (the LLM baseline trained on xLAM sits at 49).
5. **Laya's encoder can't fill arguments:** MLM loss 4.99 vs 1.37 for base ModernBERT (3.6x worse).
6. **Calibration:** ECE ~0.09-0.11 using min(tool conf, arg confs) - usable to escalate low-confidence calls.
7. **Ablation (markers only, no hard negatives):** markers alone fix same-tool parallel calls (parallel 58.5 -> 62.0);
   hard negatives drive irrelevance (75.8 -> 83.3) and parallel-multiple (49.5 -> 56.0). The +-2-3 pt moves on
   BFCL simple/multiple across these runs are within noise (n=200-400 -> ~3 pt standard error), not a real regression.

## What it cost

**$5.41 total** (account $32.96 -> $27.55) for 17 instance launches: 9 training runs, 2 benchmark jobs, 1 smoke test,
5 failed/duplicate boxes (~$0.75 of the total). Per-run table: `docs/results/costs.md`.

## Next (recommendation)

1. **Distillation** is the obvious next lever: every remaining error is an argument *value*, and parallel decoding
   is known to gain most from deterministic teacher targets. Relabel train_v2 with a strong LLM (~$7-9 on an A100).
2. **Optimised inference** (CUDA graphs / torch.compile on the fixed-shape slot pass) to show the cached mask's
   real latency advantage; today both models are launch-overhead-bound.
3. Multi-seed runs before claiming any +-3 pt BFCL difference.

## Infra lessons

- `pytorch/pytorch` image on Vast uses system Python 3.12 (PEP 668): `pip install --break-system-packages`.
- Installing the `vastai` CLI on the box breaks (Debian-owned `cryptography`); self-destroy uses Vast's REST API
  with the instance-scoped `CONTAINER_API_KEY` via `curl` instead.
- `vastai destroy instance` waits on a y/N prompt unless `-y`; the laptop watcher now passes `-y`.
- Some hosts can't reach huggingface.co or hang on image pulls; `onstart.sh` self-destroys if the bundle can't
  be fetched and `infra/bad_machines.txt` excludes known-bad hosts.
- HF ModernBERT + padding masks uses SDPA's math path; gradient checkpointing is required on 24 GB.
- Bug found and fixed: the slot plan misaligned when a zero-parameter tool call preceded another call
  (0/63k xLAM examples affected; ToolACE crashed on it). Regression test added.
