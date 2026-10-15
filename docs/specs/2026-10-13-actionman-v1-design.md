# ActionMan v1 — design spec

Date: 2026-10-13 · Status: **awaiting review** · Owner: Shreyash

## 1. Goal

A tool-calling model that is **close to a small LLM's accuracy and 10×+ faster**, built as research on one question:

> Can a Jev-style decision encoder plus a masked-diffusion argument filler produce complete tool calls (tool + arguments) in a few parallel passes instead of ~30 sequential tokens?

### Success criteria (v1)

| Metric | Target | Measured on |
|---|---|---|
| Tool selection accuracy | ≥ 85% (expect 90%+) | BFCL non-live + held-out xLAM |
| Full-call exact match (tool + all args) | ≥ 50% to continue, ≥ 75% = strong result | same |
| Accuracy gap vs our Qwen3-0.6B baseline | as small as possible; reported honestly | same data, same eval |
| Latency speedup vs baseline | ≥ 10× at batch 1 | same GPU, same serving style |
| Spend | ≤ $27.96 (balance $32.96 − $5 floor) | Vast ledger |

### Non-goals (v1)

Multi-turn / agent loops · distillation · copy head · training on Glaive/Hermes/ToolACE · production serving · publishing. All are v2 candidates (§10).

## 2. Architecture

```
input:  [TOOLS] sig_1 [T] sig_2 [T] ... [USER] message [CALL1][CALL2][CALL3][CALL4]
                                                        |
                               ModernBERT-large encoder (395M, 28 layers, 8k ctx)
                                                        |
        +-------------------------------+---------------+-------------------------------+
        | Decision head (Laya-style)    | Length head                                   |
        | each [CALLi] scores every [T] | per (call, arg): bucket                       |
        | marker + NONE -> tool or none | {absent, ≤4, ≤8, ≤16, ≤32, overflow}          |
        +-------------------------------+-----------------------------------------------+
                                                        |
   pass 1..T:  ... [ARGS] call1: q: [M]x8  k: [M]x4  call2: ...      (slot block appended)
                                                        |
               same encoder + its MLM head  ->  a token at every [M]
               re-mask slots with confidence < τ, repeat; stop early when all ≥ τ
                                                        |
output:  [ {tool, args: {name: value}, conf_tool, conf_per_arg}, ... ]  -> typed JSON
```

### 2.1 Input format — compact signatures

Tools are converted from OpenAI-style JSON schema to one line each:

```
web_search(q:str, k:int=5, site?:str) | web search
```

- Description trimmed to ~10 words; param descriptions only where the name is ambiguous.
- A `[T]` marker token precedes each signature; the decision head scores these markers.
- Measured in Stage 0: token count JSON vs signature on ModernBERT's tokenizer.

### 2.2 Decision head (tool choice)

- Copy of Laya's design: 2 small transformer layers + scorer (verified from the Laya checkpoint: `head.layers` 25M params, `scorer`), trained from scratch.
- 4 call slots `[CALL1..4]`. Each scores all `[T]` markers + a NONE option (softmax). Covers BFCL simple, multiple, parallel, parallel-multiple, and irrelevance (all NONE).
- Call slots are ordered by the order calls appear in the gold answer.

### 2.3 Length head (no hard cap)

- For each (chosen tool, argument): a 6-way bucket `{absent, ≤4, ≤8, ≤16, ≤32, overflow}`.
- `absent` handles optional args. Slot count = bucket's upper bound; unused slots are trained to `[PAD]`.
- `overflow` (> 32 tokens) → v1 marks the argument as escalated and counts it as a miss in eval (honest).
- Bucket edges are a starting point; Stage 0's length profile may move them.

### 2.4 Argument filler — masked diffusion (Mask-Predict)

- Slot block appended after the decision: for each predicted call, each present arg gets `name:` + N `[MASK]` tokens.
- One forward pass predicts every mask at once using ModernBERT's **MLM head** (vocab 50,368 — it can write any word; nothing is restricted to the input).
- Refinement: re-mask slots with confidence < τ (start τ = 0.85), re-run, up to T passes; early exit when none are below τ. Evaluated at T = 1, 2, 3.
- All arguments are typed as text slots in v1 (ints, floats, bools, enums, lists are written as text, then parsed + validated against the signature). Enum/bool via the decision head is v2.
- v1 uses the **standard bidirectional mask**: each refinement pass re-encodes the full sequence. Accuracy is identical to the cached-mask version; only speed differs.

### 2.5 Backbone choice

- Start from `answerdotai/ModernBERT-large` (MLM head intact).
- Stage 1 test: graft base ModernBERT's MLM head onto Laya's encoder and compare MLM loss. If within ~5% of base, start from Laya's encoder instead (it has seen decision training). Decision recorded in the run log.

## 3. Training

### 3.1 Data

| Split | Source | Size |
|---|---|---|
| Train | Salesforce/xlam-function-calling-60k (gated, access verified) | ~58k after holding out 2k |
| Dev / test | held-out xLAM | 1k / 1k |
| Eval | BFCL non-live (simple, multiple, parallel, parallel-multiple, irrelevance) | as published; never trained on |

Irrelevance negatives: v1 synthesises ~5k "no tool" examples by pairing xLAM queries with tool lists that don't contain the needed tool (target: all NONE).

### 3.2 Objectives (summed)

1. **Decision**: cross-entropy per call slot over `[T]` markers + NONE.
2. **Length**: cross-entropy over the 6 buckets per (gold call, arg).
3. **Slots**: masked-diffusion loss. Each example, sample a mask ratio r ~ U(0,1] and mask that fraction of argument tokens (always ≥ 1); predict them. Gold buckets set the slot count during training (teacher forcing).

### 3.3 Settings (starting point, tuned only in ablations)

bf16 · AdamW · LR 5e-5 encoder / 1e-3 heads · warmup 5% · cosine · 3 epochs · max seq 1024 · sequence packing (ModernBERT unpadding) · effective batch ~64k tokens · seed fixed.

### 3.4 Baseline

Qwen3-0.6B, full fine-tune on the **same** xLAM split, Hermes-style JSON tool-call output, 2–3 epochs, same tool descriptions (JSON schema form). Evaluated with the same scorer.

## 4. Evaluation

- **Accuracy**: BFCL AST checker semantics (tool name + argument values with type coercion), per category. Report tool accuracy, per-argument accuracy, full-call exact match.
- **Passes curve**: full-call accuracy and latency at T = 1, 2, 3.
- **Calibration**: ECE of tool confidence and per-arg confidence.
- **Latency**: batch 1, p50/p95 over 500 BFCL prompts, same GPU, both models in plain PyTorch (fair comparison). Baseline additionally under vLLM if budget allows, reported separately.
- Every result row records: checkpoint id, run id, GPU, $ spent.

## 5. Stages, caps, gates

Balance at start: **$32.96**. Floor: **$5** (never touched). Spendable: **$27.96**.

| Stage | Where | GPU | Cap | Work | Gate to continue |
|---|---|---|---|---|---|
| 0 Build | laptop | — | $0 | Converter, length profile, tokenisation, training + eval code run end-to-end on a tiny random model on CPU, watchers | Tiny model trains and evals end-to-end |
| 1 Smoke | Vast | cheapest 24 GB card | $1.50 | Overfit 500 examples; measure tokens/s; Laya MLM-graft test; verify self-destroy + HF upload path | Loss → ~0; all later caps recomputed from measured speed |
| 2 v1 | Vast | 4090/5090 on-demand | $3 | xLAM 3 epochs | Tool ≥ 85%, full call ≥ 50% on dev; else debug, no new spend |
| 3 Baseline + eval | Vast | same class | $3 | Qwen3-0.6B SFT; BFCL for both; latency; passes 1/2/3 | Headline table |
| 4 Ablations + attention mask | Vast | 4090 (interruptible for ablations) | $8 | Mask-ratio schedule, τ, bucket edges, Laya vs base encoder; **block-causal cached mask** trained directly on xLAM, compared vs v1 on accuracy + measured latency; short generic MLM adaptation only if accuracy drops > 2 pts | Know what matters; full speed story measured |
| 5 Upgrade | Vast | per choice | $10 | One of: distillation with a ~32B teacher on an A100 80GB (args garbled) · more data: Glaive + Hermes + ToolACE (unseen tools fail). Chosen with you after Stage 4 | v2 numbers |
| Overhead | — | — | $2.46 | idle, mistakes, bandwidth | — |

Stage 5 is not started without your go.

## 6. Infrastructure & credit safety

### 6.1 Checkpoints

Private Hugging Face repo under `shing-dev` (token verified: read + write; nothing public). Box uploads final bf16 weights + logs at job end; resume checkpoints (with optimizer state) stay on the box during the run. Laptop pulls models on demand. Publishing is a separate later step.

### 6.2 Four safety layers

| Layer | Where | Rule |
|---|---|---|
| L1 job wrapper | box | `timeout <stage cap>`; on any exit → upload to HF → `vastai destroy instance $CONTAINER_ID` (instance-scoped key, per Vast docs) |
| L2 watchdog | box | GPU util < 5% or heartbeat stale for **10 min** → upload → destroy |
| L3 watcher | laptop, Windows scheduled task, every 2 min | destroy any instance: over its stage cap, not in the ledger, or in stopped state; destroy all if balance < $5; desktop notification on every action |
| L4 account | Vast billing | user confirms no auto-top-up (account shows a payment method on file) |

- **Destroy, never stop**: stopped instances still bill storage.
- **Ledger** (`infra/ledger.jsonl`): instance id, stage, start, cap, end, cost. L3 reads it.
- **Host filter**: reliability > 98%, low internet $/TB, best DLPerf/$, CUDA ≥ 12.4, disk 40 GB.
- **Data upload**: pre-tokenised shards from the laptop (MB-scale), no preprocessing on the clock.

## 7. Repository layout

```
ActionMan/
  env.manifest          HF_TOKEN (names only)
  docs/specs/           this spec
  src/actionman/
    data/convert.py     OpenAI JSON -> signatures + slot targets
    data/profile.py     token-count and arg-length profiling
    data/tokenize.py    shards for upload
    model.py            encoder + decision head + length head + MLM slot filler
    train.py            objectives, packing, checkpointing
    infer.py            decision -> slots -> T passes -> typed JSON
  baseline/train_qwen.py
  eval/bfcl.py          AST scorer, per-category metrics, ECE
  eval/latency.py
  infra/
    job.sh              L1 wrapper
    watchdog.py         L2
    watcher.py          L3 (laptop)
    launch.py           host search + create + ledger entry
    ledger.jsonl
  tests/                CPU tests on a tiny random model
```

## 8. Testing

- Converter: round-trip tests (signature → parsed back to the same schema) on 200 xLAM tools.
- Model: CPU tests on a ~1M-param random ModernBERT config — shapes, masking, loss decreases on 20 examples, inference produces valid JSON.
- Infra: dry-run mode for `launch.py` and `watcher.py` (no API mutations); one real $0.10 self-destroy test in Stage 1.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Parallel slots garble longer args | length buckets, T up to 3, distillation in v2 |
| Same tool called twice / call order mismatch | order-insensitive matching in the scorer for parallel categories |
| Laya encoder lost MLM ability | Stage 1 graft test; fall back to base ModernBERT |
| Measured speed slower than estimated | Stage 1 recomputes all caps before spending further |
| Baseline unfairly weak or strong | same data, epochs within ±1, same eval code, both plain PyTorch |
| Credits leak | L1–L4, $5 floor, ledger |

## 10. Deferred to v2 (only if v1 numbers justify)

Distillation from a ~32B model · copy (pointer-generator) head · enum/bool via decision head · Glaive/Hermes/ToolACE/When2Call data · multi-turn agent loop · CUDA graphs / optimised serving · publishing to the Hub.
