# ActionMan benchmark + release plan (2026-10-15)

Status: draft for approval. Nothing is committed; results are handed back for review.
No LLM hand-off anywhere in this plan (rejected: it breaks the one-small-model promise).

## Goal

1. **Publishable numbers** for the release candidate, measured on the official BFCL v4 harness and one shared latency box.
2. **A release package:** HF model repo, an installable `predict()` API, a model card, and a tech report outline.

Release candidate: `s16-v3-5epb` (v3 recipe, `--v3-mask legacy`, 5 epochs on train_v2), private HF repo `runs/s16-v3-5epb/m/`.
It is ModernBERT-large (395M) plus heads, so about 0.4B parameters.

| what we know today (own eval, 2,340 prompts, our scorer) | value |
|---|---|
| full-call accuracy, T=3 | 79.7% (seed noise ~1.5 pts) |
| tool-name accuracy | 95.1% |
| BFCL irrelevance (v3 subset) | 80.8% |
| same-data Qwen3-0.6B SFT | 86.5% |
| latency, 4090, batch 1 | ActionMan compiled p50 16.7 ms vs Qwen HF generate 1,633 ms (Qwen not optimised) |
| laptop CPU, fp32, ~200-token prompt, 4 tools | ~2.5-3.8 s per call (a real risk) |

**"Done" means:**
- Table 1: BFCL v4 Non-Live / Live / Hallucination, for ActionMan and 6 baselines, from one harness.
- Table 2: GPU latency (p50/p95/mean plus accuracy on the same prompts) against a tuned vLLM, on one box.
- Table 3: CPU latency against llama.cpp, with the same CPU and threads.
- The model card and API are ready to publish once the licence decision is made.

**We will NOT claim:**
- a BFCL Overall score. Unrun categories count as 0, and Agentic plus Multi-turn are 70% of Overall.
- SOTA, or "beats LLMs" on accuracy. Our own Qwen SFT is 6.8 pts higher.
- multi-turn or agentic support. ActionMan is single-turn only, and we say so.
- the 90x ratio. It came from an unoptimised baseline. Only the ratio measured against tuned vLLM is used.
- a CPU speed advantage, unless Table 3 shows one.

## Workstream A: accuracy on the official harness (BFCL v4)

**Setup:** `pip install bfcl-eval==2026.3.23`, in its own uv venv together with vLLM (last night's lesson: `VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`).

**ActionMan handler (new files, built and smoke-tested on the laptop at $0):**
- `bench/am_server.py`: a tiny localhost HTTP server in ActionMan's own env (transformers 5.19.0). It loads the checkpoint once (`evaluate.load_model`, bf16) and calls `infer.predict(T=3, tau=0.85)`. This keeps the bfcl/vLLM venv and ActionMan's pinned deps apart.
- `bench/bfcl_actionman.py`: the handler. It subclasses `BaseHandler` directly (simpler than `OSSHandler`, whose `spin_up_local_server` is @final and expects vLLM/SGLang). This is untested; the fallback is `OSSHandler` with overrides of `_query_prompting` -> (response, latency) and `_parse_query_response_prompting`. It POSTs (query, tools) to the server and implements `decode_ast` -> `[{func: {param: val}}]`.
- `bench/bfcl_register.py`: adds `actionman-s16` to `constants/model_config.py` and `constants/supported_models.py` in the installed package. The model is non-public, so it is scored locally and not listed.

**Prompt flattening (live data has system and prior messages; `from_bfcl` keeps only the last user message):**
- Default: join every message into one query, `System: ... / User: ... / Assistant: ...`, with the last user turn at the end. If it is too long, truncate from the front so it fits `MAX_LEN_A` = 896 tokens.
- Ablation: last-user-only on `live_simple` + `live_multiple`. Report both and use the better one everywhere, stated in the card.

**Tool conversion:**
- BFCL function docs go through the same path as `schema.from_bfcl` / `norm_type`. Unknown types (Java `HashMap`, JS `any`) map to `any`.
- No-call output returns plain text, which `decode_ast` turns into `[]`. The smoke test checks that the checker counts this as a correct abstention on 20 irrelevance and 20 relevance items.
- Prompts that don't fit, or that need more than `MAX_CALLS` = 4 calls, are scored wrong (not dropped), and their count is reported.

**Categories (3,641 prompts):**

| group | categories (n) | report |
|---|---|---|
| Non-Live | simple_python 400, multiple 200, parallel 200, parallel_multiple 200 | main |
| Non-Live, other languages | simple_java 100, simple_javascript 50 | separate column; ActionMan never saw Java/JS types |
| Hallucination | irrelevance 240, live_irrelevance 884, live_relevance 16 | main |
| Live | live_simple 258, live_multiple 1053, live_parallel 16, live_parallel_multiple 24 | main; n=16/24 are flagged as noisy |
| not run | multi-turn 4x200, web_search 2x100, memory 3x155 | "unsupported" |

- Sub-scores come from `bfcl evaluate --partial-eval` exactly as printed. We also add our own clearly labelled "Non-Live, Python only" column.

**Validation gate (first step on the box, blocking):**
- Run `bfcl generate --model Qwen/Qwen3-0.6B-FC --backend vllm` on the same categories, then `bfcl evaluate`.
- Pass: Non-Live within 1.0 pt of the leaderboard's 71.79, and Irrelevance within 2 pts of 80.84. Live (56.62) is a secondary check, ±2.
- Fail: the job stops itself and writes `GATE_FAIL`. No other harness numbers are trusted, and nothing else is launched until it is debugged (bfcl/vLLM version, dtype, chat template).

**Leniency gap (our scorer vs official):**
- `bench/to_bfcl_format.py` exports our 2,340 prompts (`data/prepared/test.jsonl` + `bfcl.jsonl`) as BFCL-format question/function/possible_answer files, registered as an extra category in our installed copy.
- ActionMan predictions are then scored twice: `scoring.py` (lenient: numeric strings, "true", recursive dicts) and the official strict AST checker.
- Gap = lenient minus strict. Before blaming the model, hand-check 50 strict-only failures for xLAM label/type noise (xLAM gold is exact-value, built by `gold_from_calls`).
- Fallback if registration takes over 1 h: call `bfcl_eval`'s AST checker directly from a script over the dumped predictions.
- Needs a `--dump` flag on `evaluate.py` (write every prediction, not only errors).
- Also note that our v3 subset differs from v4 in a handful of items (2 parallel + 2 irrelevance questions, 4 answers). Per-item diffs on the overlap separate "data changed" from "scorer stricter".

## Workstream B: baselines on the same harness and box

| model | params | licence | leaderboard NL / Live / Irr | why |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B (FC) | 752M | Apache-2.0 | 71.79 / 56.62 / 80.84 | validation anchor; nearest Apache LLM |
| our Qwen3-0.6B SFT (`runs/s5-qwen-v2/qwen`) | 752M | (not released) | - | same data as ActionMan: the controlled comparison |
| ibm-granite/granite-4.0-350m | 352M | Apache-2.0 | 67.92 / 46.11 / 60.84 | closest size to ActionMan |
| google/functiongemma-270m-it (FC) | 268M | Gemma terms, gated | not listed (card: simple 61.6, irr 73.7) | smallest purpose-built caller; key speed rival |
| MadeAgents/Hammer2.1-0.5b | 494M | CC-BY-NC-4.0 | 65.98 / 54.63 / 80.79 | strong irrelevance at 0.5B |
| Qwen/Qwen3-1.7B (FC) | 2.03B | Apache-2.0 | 82.92 / 74.61 / 76.54 | "what 4x the size buys" ceiling |

- **Skipped** to save time and money: xLAM-2-1b/3b, Hammer2.1-1.5b, Arch-Agent-1.5B. They are larger and add no new story.
- **needle3** (~121M, Apache-2.0) has no handler or numbers. Optional only if a handler takes under 1 h on the laptop; otherwise listed as "not evaluated".
- Our SFT uses Hermes `<tool_call>` JSON with Qwen's template, so it runs through the existing Qwen FC handler via `--local-model-path`.
- **Own eval for public models:** run them on the exported 2,340-prompt category (official checker). Caveats:
  - they are out of distribution on xLAM labelling conventions ("Paris" vs "Paris, France");
  - Hammer and xLAM-family models may have trained on xLAM-60k, so our xLAM test split may be contaminated for them.

  BFCL stays the primary table.
- Public leaderboard numbers are shown next to ours as a cross-check, never mixed into the same column.

## Workstream C: latency (the headline claim, so it must be airtight)

**Protocol (all arms, same RTX 4090 box, same session, batch 1):**
- **Prompts:** 500 fixed prompts, stratified over BFCL v4 non-live python + live. The IDs are frozen in `bench/latency_ids.json` and published.
- **Timed span:** end to end, from (query, OpenAI-style tools) to a parsed list of calls. It includes templating/tokenisation and output parsing, and excludes model load.
- **Warm-up:** the first 20 prompts plus every compile/CUDA-graph capture are excluded.
- **Drift check:** each arm runs twice in ABAB order to detect host drift. Each arm reports p50 / p95 / mean, output tokens, and full-call accuracy on the same 500 (a fast wrong answer is not a win).
- **Code:** extend `bench/latency.py` (do not fork it) with `--ids`, `--arms`, `--threads`, and vLLM/llama.cpp arms. Record GPU, driver, CPU model, and vLLM/llama.cpp/torch versions in the JSON.

| arm | engine | notes |
|---|---|---|
| ActionMan eager, T=3 (and T=1) | torch bf16 | existing path |
| ActionMan compiled, T=3 (and T=1) | `cached.make_static` + CUDA graphs | release default |
| Qwen3-0.6B SFT | vLLM, in-process `LLM.chat`, bf16, CUDA graphs on, greedy, max_tokens 256 | headline rival; tuned: cudagraph sizes incl. 1, max_num_seqs 1 also tried, best of both kept |
| Qwen3-0.6B public FC | vLLM, same settings | its output format is longer |
| functiongemma-270m | vLLM, same settings | smallest LLM rival |
| Qwen3-0.6B SFT | llama.cpp GGUF Q8_0, full GPU offload | prebuilt CUDA wheel; dropped if not working in 20 min |
| Qwen3-0.6B SFT | HF `generate` (existing `bench_qwen`) | reference only, not in headline |

- **Fairness:** prefix caching is off for both sides in the headline (ActionMan's catalog-token caching is also off). A "warm catalog" variant for both sides is an optional row.
- **Output-length dependence:** bucket the 500 prompts by gold call count (1 / 2 / 3+) and by output tokens. Plot latency vs output size. We expect LLM latency to grow with calls/args and ActionMan to stay roughly flat; we report it either way.
- **Throughput (optional):** batch 32 on vLLM is trivial. ActionMan has no batched `predict`, so its side is skipped unless one exists. We will not compare batch-32 LLM against batch-1 ActionMan.

**CPU (same CPU, fixed threads):**
- **Vast box CPU:** `taskset` to 8 cores. ActionMan fp32 with `torch.set_num_threads(8)` vs llama.cpp Q8_0 (Qwen3-0.6B SFT, functiongemma) with `-t 8`. 200 of the 500 prompts.
- **Laptop CPU ($0):** the same comparison with Windows llama.cpp binaries, on AC power, the same thread count for both, and the CPU model recorded. This is the "consumer CPU" row. Include a 10-prompt torch-profiler breakdown of where ActionMan's 2.5-3.8 s goes.
- **Follow-ups (NOT in this plan):** int8 / ONNX Runtime export, a ModernBERT-base variant, reusing the context cache across refinement passes on CPU.

## Workstream D: extras (laptop, $0, from dumped predictions)

- **Seed noise:** the v3 seeds give ~1.5 pts. Report it as the band on our eval, plus a 95% bootstrap CI per BFCL category (n=16-24 live categories will show huge CIs, and that is the point).
- **Calibration:** ECE from `evaluate.ece` (min of tool/arg confidence), plus a reliability diagram, on our eval and BFCL v4.
- **Accuracy vs coverage:** ActionMan abstains below threshold theta. Report the curve and the points at 100/90/80% coverage. This is a "say I'm not sure" knob for the API, not a hand-off.
- **Error breakdown:** `errclass.py` classes (wrong tool, span length, reasoning, list format, truncated, garbled) per BFCL v4 category, plus the count of prompts skipped for length or more than 4 calls.

## Workstream E: wrapping for release

**HF model repo** (`shing-dev/actionman-0.4b`, private until the licence gate clears):
- `model.safetensors` (converted from `model.pt`, fp32 + bf16).
- `config.json`: the encoder config plus `run.json` flags (v3, cached, markers), `MAX_CALLS` 4, and token limits 896/1024.
- Tokenizer with the added special tokens.
- `README.md` model card covering:
  - intended use;
  - Tables 1-3;
  - limits: single-turn, English, at most 4 calls, lists of up to 7 items, no reasoning or unit conversion, the CPU caveat;
  - training data with each licence;
  - training cost.

**API:** `src/actionman/api.py`, installable via the existing `pyproject.toml`:
```python
am = ActionMan.from_pretrained("shing-dev/actionman-0.4b", device="cuda", compile=True)
am.predict(query, tools, T=3, threshold=None)
# -> [{"name", "arguments", "confidence", "arg_confidences"}], abstained: bool
```
- `tools` accepts OpenAI `{"type":"function","function":{...}}` or bare JSON-schema functions; the reverse of `to_openai_tool`.
- Test: identical outputs to `evaluate.py` on 50 BFCL items (`tests/test_api.py`).

**Demo (optional):** a Colab notebook. A free-CPU HF Space would show the slow CPU path, so not until the CPU work is done.

**Tech report / blog outline:**

| section | contents |
|---|---|
| 1 Intro | tool calling as structured prediction; why non-autoregressive |
| 2 Method | pass A, slot block, copy/option/list slots, mask-predict, cached context, CUDA graphs (one diagram) |
| 3 Data + licences | source table with licence and generator |
| 4 Accuracy | Table 1 (BFCL v4 sub-scores, params, licence), Table 4 per-category incl. Java/JS, Table 5 own eval lenient vs strict |
| 5 Latency | Table 2 GPU, Figure latency vs output size, Table 3 CPU |
| 6 Analysis | calibration, coverage curve, error classes, seed noise, ablations (v2 -> v3, T1 vs T3, eager vs compiled, 3 vs 5 epochs) |
| 7 Limitations | single-turn, no reasoning, CPU, licence, lenient-scorer history |
| 8 Repro | scripts, versions, prompt IDs, cost |

The blog is sections 1, 4 (Table 1 only), 5 (figure + Table 2) and 7, at about 1,000 words.

## Licence gate (blocking for public weights; DECISION FOR THE USER)

**The problem:**
- train_v2 contains xLAM-60k. It is CC-BY-4.0, but gated, and its card says "This release is for research purposes only". It was generated by DeepSeek-V2-Chat and Mixtral-8x22B.
- Our hard negatives were also built from xLAM queries (the reason When2Call was dropped).
- Hermes v1 (Apache-2.0, generator unknown) and ToolACE (Apache-2.0, "a frontier LLM") are cleaner, but not risk-free.

| option | what | cost | consequence |
|---|---|---|---|
| (a) | release s16-v3-5epb under research / non-commercial terms | $0 | fastest; limits adoption; still not legal advice |
| (b) | retrain the same recipe without xLAM: ToolACE + Hermes + Apache train_v4 extras (Synth-APIGen, Glaive v2, Toucan) + negatives rebuilt from non-xLAM queries; measure the drop on BFCL v4 | ~$1.5-2 (5-epoch run was 2.1 h on a 4090 at ~$0.6/h, plus eval) / cap $2.40 | Apache-style release possible; expect some drop (xLAM is most of train_v2; train_v4 extras did not help at 3 epochs) |
| (c) | get a legal read on (a) vs Apache | time, maybe money | slowest; may be needed anyway before commercial use |

- Benchmarking (A-D) is not blocked; only publishing weights is.
- Our xLAM test split is not redistributed under any option.

## Runs and budget

Credit $17.17. `launch.py` refuses any launch whose worst case crosses the $12 floor; the laptop watcher kills everything below $5.
Every box gets the usual layers:
- L1: `box_job.sh` hard timeout plus self-destroy (10 retries);
- L2: `watchdog.py`;
- L3: laptop `watcher.py`, re-registered for the night and removed after;
- an entry in the ledger (`infra/ledger.jsonl`).

| run | box | job | expected | cap |
|---|---|---|---|---|
| b1-bfcl | RTX 4090, disk 80 GB, `--max-dph 0.6` | gate -> ActionMan all categories + flattening ablation -> our SFT -> 5 public baselines -> own-eval export for all 7 -> strict rescore | ~2.5 h, $1.25 | 4 h, $2.40 |
| b2-latency | RTX 4090, disk 60 GB, `--max-dph 0.6` | all GPU arms (ABAB) + box-CPU arms | ~1.5 h, $0.75 | 2.5 h, $1.50 |
| laptop-cpu | laptop | CPU arms + profiler | $0 | $0 |
| stuck-box allowance | - | Vast "loading" hosts (cost $0.76 last night) | $0.30 | $0.50 |
| **night 1 total** | | | **~$2.30** | **$4.40 (credit stays >= $12.77)** |
| b3-clean (optional, night 2, only if licence option b) | RTX 4090 | retrain without xLAM, 5 epochs, + own eval + BFCL v4 run | ~$1.60 | $2.40 |

- Night 2 fits above the $12 floor only if night 1 spends at most ~$2.75. Otherwise it needs a top-up or a user-approved floor change. I will not lower the floor myself.

## Schedule

| when | what | $ |
|---|---|---|
| day 0 (laptop) | handler + server + register script, `--dump`, BFCL export, latency harness arms; CPU smoke on 20 items per category; `launch.py --dry-run` | 0 |
| night 1 | b1-bfcl and b2-latency in parallel; laptop CPU benchmark | ~2.30 |
| day 1 (laptop) | workstream D, tables, framing decision, licence decision | 0 |
| night 2 (optional) | b3-clean | ~1.60 |
| day 2 (laptop) | API, safetensors, model card, report draft | 0 |

## Decision triggers

- **Harness gate fails** (Qwen3-0.6B FC Non-Live is off by more than 1 pt): stop, debug, launch nothing else.
- **GPU latency:** ratio = tuned-vLLM Qwen3-0.6B SFT p50 / ActionMan compiled p50, on the 500 prompts.
  - **>= 5x:** use the "fast tool caller" framing; latency leads the card.
  - **3-5x:** lead with absolute ms plus accuracy, and keep the ratio in the text with caveats.
  - **< 3x:** use the "architecture" framing (non-autoregressive tool calling); speed is a secondary result.
- **functiongemma-270m on vLLM is both faster and more accurate than ActionMan:** no speed claim at all.
- **CPU:** if ActionMan's p50 is higher than llama.cpp Qwen3-0.6B Q8_0 at the same threads, the card says "fast on GPU, not on CPU" and the CPU follow-ups get their own plan.
- **Leniency gap > 3 pts:** every ActionMan number in the release uses the official strict checker, including our own eval.
- **ActionMan below granite-350m on official Non-Live (Python):** position on latency and calibration, not accuracy.

## Risks and mitigations

| risk | mitigation |
|---|---|
| bfcl-eval version drift vs leaderboard | validation gate; pin `bfcl-eval==2026.3.23` and vLLM version |
| handler mis-scores no-call / relevance | smoke-test 20 + 20 items by hand on the laptop |
| live prompts too long (896-token pass A) or more than 4 calls | score as wrong, report the count; flattening ablation |
| dependency clash (bfcl/vLLM vs transformers 5.19.0) | separate venvs, ActionMan behind `am_server.py` |
| vLLM install on runtime image | uv venv + DeepGEMM/FlashInfer env flags (known fixes) |
| llama.cpp CUDA needs nvcc | prebuilt CUDA wheel; else drop the GPU llama.cpp arm (vLLM is the stronger rival) |
| "you tuned the baseline badly" | publish the script, configs, versions and prompt IDs; report baseline accuracy next to its latency |
| noisy host or shared vCPUs | same box for all arms, ABAB order, pinned threads, laptop CPU as a second machine |
| gated models (functiongemma) | user accepts the Gemma terms on HF before night 1 |
| contamination of our xLAM test for Hammer | BFCL is the primary table; flag it in the own-eval table |
| stuck Vast hosts | `bad_machines.txt`, `timeout 900` on the bundle, allowance in the budget |

## Approval needed

1. **Budget:** night 1 expected ~$2.30, cap $4.40 (credit $17.17 -> at least $12.77). Night 2 (optional) ~$1.60 / cap $2.40.
2. **Licence option:** (a) research / non-commercial, (b) clean retrain (night 2), or (c) legal read. Weights stay private until you choose.
3. **Baseline set:** Qwen3-0.6B FC, our SFT, granite-4.0-350m, functiongemma-270m, Hammer2.1-0.5b, Qwen3-1.7B FC; needle3 only if cheap. Also accept functiongemma's gated terms on the HF account.
4. **Repo name:** the public repo name (`shing-dev/actionman-0.4b` proposed), and whether a Colab demo is wanted.
