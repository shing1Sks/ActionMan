# ActionMan v1 Implementation Plan

> **For agentic workers:** executed natively (superpowers:executing-plans) in an away session approved by Shreyash on 2026-10-13. Steps use checkbox syntax. **No commits** — the repo is git-initialised on `dev` with everything left uncommitted for review.

**Goal:** Build and evaluate ActionMan v1 (decision encoder + masked-diffusion argument filler) against a same-data Qwen3-0.6B baseline, within the Vast budget.

**Architecture:** ModernBERT-large encoder. Pass A reads `[tools][user][CALL]x4` → decision head (each `[CALL]` scores `[T]` markers + NONE) and length head (bucket per arg). Pass B appends a slot block of `[MASK]`s → MLM head fills all at once → re-mask low-confidence → repeat (T ≤ 3).

**Tech Stack:** Python 3.12, PyTorch, HF transformers (ModernBERT, Qwen3), huggingface_hub, vastai CLI, pytest.

**Spec:** `docs/specs/2026-10-13-actionman-v1-design.md`

## Global Constraints

- Balance $32.96 at start; **$5 floor never touched**; stage caps are soft (agent judgement), the watcher's hard limits are 1.5× each stage's cap.
- Never commit, merge, or push. Never print or log secret values (`HF_TOKEN` read only by scripts).
- Destroy instances, never leave them stopped.
- Checkpoints only to the **private** HF repo `shing-dev/actionman`; nothing public.
- Max 4 calls per example; length buckets `{absent, ≤4, ≤8, ≤16, ≤32, overflow}`; τ = 0.85; T ∈ {1,2,3}.
- Max sequence length 1024 (pass B). Examples that don't fit after dropping descriptions are skipped and counted.

## Review Focus

1. Gold call with **more than 4 calls** (xLAM 0.6%, BFCL parallel) → training drops it; eval counts it as a miss (never crashes).
2. Argument values that are **lists / dicts / floats / bools** → serialised as JSON text, parsed back by declared type; `3` vs `3.0` must match.
3. **Optional params** omitted by the model → correct if gold allows omission (`""` in acceptable values) or equals default.
4. **Irrelevance** prompts (no tool applies) → model must emit zero calls; a call is a miss.
5. A Vast box that **hangs, crashes, or outlives its cap** → destroyed by L1/L2/L3 with outputs uploaded first.

## File map

```
src/actionman/schema.py     Tool/Param/Call dataclasses; xLAM + BFCL converters; signature rendering; value<->text
src/actionman/features.py   tokenizer setup (special tokens), pass-A / pass-B sequence builders, length buckets
src/actionman/data.py       Dataset + collate (random mask ratio), train/dev/test split, irrelevance negatives
src/actionman/model.py      ActionMan nn.Module: encoder + MLM head + decision head + length head; losses
src/actionman/infer.py      predict(): pass A -> slot block -> mask-predict -> typed calls with confidences
src/actionman/scoring.py    BFCL-style AST matcher (order-insensitive, string normalisation, numeric coercion)
src/actionman/train.py      training loop, heartbeat, checkpoint, dev eval
src/actionman/evaluate.py   run model over xLAM test + BFCL categories, metrics JSON, passes curve
src/actionman/profile.py    arg-length + context-length profile report
baseline/qwen.py            Qwen3-0.6B SFT + predict (same scorer)
bench/latency.py            batch-1 latency for both models
infra/vast.py               thin wrapper over vastai CLI (--raw JSON)
infra/launch.py             bundle -> HF, pick offer, create instance, ledger entry
infra/box_job.sh            L1: setup, watchdog, timeout run, upload, self-destroy
infra/watchdog.py           L2: 10-minute stale-heartbeat rule on the box
infra/watcher.py            L3: laptop watcher (scheduled task, every 2 min)
tests/                      CPU tests on a tiny random ModernBERT
```

---

### Task 1: Schema, converters, value round-trip

**Files:** `src/actionman/schema.py`, `tests/test_schema.py`

**Produces:** `Param(name, type, optional, default, description)`, `Tool(name, description, params)`, `Call(name, args)`, `Example(query, tools, calls, id, source)`; `from_xlam(row) -> Example`; `from_bfcl(q_row, answer_row|None, category) -> (Example, gold)`; `signature(tool, desc_words=8) -> str`; `value_to_text(v) -> str`; `text_to_value(s, type) -> Any`; `to_openai_tool(tool) -> dict`.

- [ ] Tests: xLAM row 0 converts to 1 tool / 2 calls; type `"str, optional"` → `type="str", optional=True`; BFCL `required` drives `optional`; `signature()` of a 2-param tool equals the exact expected string; round-trip `text_to_value(value_to_text(v), t) == v` for `"ramen"`, `3`, `2.5`, `True`, `[3, 5]`, `{"a": 1}`.
- [ ] Implement; tests pass.

### Task 2: Features (tokenizer + sequence builders)

**Files:** `src/actionman/features.py`, `tests/test_features.py`

**Consumes:** Task 1. **Produces:** `load_tokenizer(name) -> tok` (adds `[T] [CALL] [ARGS] [USER]`); `BUCKETS=[0,4,8,16,32]`, `bucket_of(n_tokens) -> int (0..5)`; `build_pass_a(tok, ex) -> dict(input_ids, t_pos, call_pos, arg_pos[(tool_j, param_k)], dec_labels[4], len_items[(call_i, tool_j, param_k, bucket)])`; `build_pass_b(tok, ex, calls, buckets) -> dict(input_ids, slot_pos[[...]], slot_targets[[...]], slot_index[(call_i, param_k)])`.

- [ ] Tests: `[CALL]` count is 4 and `t_pos` points at `[T]` tokens; dec labels = tool index for gold calls, NONE (= n_tools) after; a 3-token value gets bucket 1 (≤4) and 4 slots with 1 PAD target; a 40-token value gets bucket 5 (overflow) and no slots; absent optional param gets bucket 0.
- [ ] Implement; tests pass.

### Task 3: Model + training step

**Files:** `src/actionman/model.py`, `src/actionman/data.py`, `tests/test_model.py`

**Produces:** `ActionMan.from_encoder(name_or_config, tok)`; `forward_a(batch) -> (dec_logits[B,4,Jmax+1], len_logits[N,6])`; `forward_b(batch) -> slot_logits[M,V]`; `loss(batch) -> (total, parts)`; `collate(examples, tok, mask_ratio_sampler)`.

- [ ] Test: tiny config (hidden 64, 2 layers) trains 60 steps on 16 examples; total loss drops by > 50%.
- [ ] Implement; test passes.

### Task 4: Inference (mask-predict)

**Files:** `src/actionman/infer.py`, `tests/test_infer.py`

**Produces:** `predict(model, tok, ex, T=2, tau=0.85) -> list[PredCall(name, args, conf_tool, conf_args, escalated)]`, `predict_batch`.

- [ ] Test: after overfitting the tiny model on 8 examples, `predict` returns the gold calls for ≥ 6/8; never raises on an example whose tool list is empty of matches.
- [ ] Implement; test passes.

### Task 5: Scoring + evaluation

**Files:** `src/actionman/scoring.py`, `src/actionman/evaluate.py`, `tests/test_scoring.py`

**Produces:** `match_calls(pred: list[Call], gold: list[dict[name, dict[arg, list[acceptable]]]]) -> bool`; `evaluate(model_fn, examples) -> metrics dict` (per-category accuracy, tool-name accuracy, per-arg accuracy).

- [ ] Tests: order-insensitive parallel match; `"New York"` vs `"new york"` match; `3` vs `3.0` match; omitted optional with `""` acceptable passes; extra unknown arg fails; irrelevance with zero calls passes.
- [ ] Implement; tests pass.

### Task 6: Data prep + profile

**Files:** `src/actionman/profile.py`, `src/actionman/data.py` (split + negatives), output `data/prepared/{train,dev,test}.jsonl`, `data/prepared/bfcl.jsonl`, `docs/results/profile.md`

- [ ] Split xLAM 58k/1k/1k (seed 0); add 5k irrelevance negatives to train, 200 to dev/test; convert BFCL simple/multiple/parallel/parallel-multiple/irrelevance.
- [ ] Profile: context tokens (signature vs JSON), arg-length buckets, % overflow, % > 4 calls, % over 1024 tokens. Write report.

### Task 7: Training script + full CPU dry run

**Files:** `src/actionman/train.py`, `tests/test_train_cli.py`

- [ ] `python -m actionman.train --data data/prepared --out runs/x --tiny --max-steps 20` runs end-to-end on CPU: writes heartbeat, checkpoint, dev metrics JSON.
- [ ] Real flags: `--encoder answerdotai/ModernBERT-large --epochs 3 --lr 5e-5 --head-lr 1e-3 --tokens-per-batch 32768 --bf16`.

### Task 8: Baseline + latency

**Files:** `baseline/qwen.py`, `bench/latency.py`

- [ ] `baseline/qwen.py train|predict` with Qwen3 chat template (`tools=`, `enable_thinking=False`), loss on assistant tokens only; parse `<tool_call>` JSON; CPU dry run with `--tiny` (random 2-layer Qwen3 config).
- [ ] `bench/latency.py --model actionman|qwen --n 300`: batch 1, `torch.cuda.synchronize`, p50/p95; ActionMan for T=1,2,3.

### Task 9: Infra + safety layers

**Files:** `infra/vast.py`, `infra/launch.py`, `infra/box_job.sh`, `infra/watchdog.py`, `infra/watcher.py`, `infra/ledger.jsonl`

- [ ] `launch.py --dry-run` prints chosen offer and the exact create command (token redacted) without creating anything.
- [ ] `watcher.py --dry-run` lists what it would destroy given the current account state.
- [ ] Register `ActionManWatcher` scheduled task (every 2 min, `pythonw`).

### Task 10: Vast stages

- [ ] Stage 1 (3090-class): smoke overfit, throughput, Laya MLM-graft test, verify self-destroy + upload. Recompute caps.
- [ ] Stage 2 (A100 40GB / 4090): v1 3 epochs → dev metrics → gate (tool ≥ 85%, full call ≥ 50%).
- [ ] Stage 3: Qwen baseline (parallel box), BFCL + xLAM-test eval both, latency, passes curve → `docs/results/results.md`.
- [ ] Stage 4: ablations + block-causal mask (if time/budget).
- [ ] Stage 5: upgrade per results.
- [ ] Hand-back: KB `log.md` entry, results table, ledger, `To verify` list.
