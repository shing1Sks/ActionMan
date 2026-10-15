# ActionMan v3 Implementation Plan (overnight 2026-10-14 -> review 2026-10-15 11:00)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking. **No commits** (user rule: never commit/merge/push without
> explicit approval); every "commit" step in the template is replaced by "leave uncommitted, note in log".

**Goal:** Close the argument-value gap (ActionMan v2 75.9% vs Qwen3-0.6B 86.5% full-call) without giving up the
~20x speed, and measure honestly how close confidence-based escalation gets to the LLM.

**Architecture:** Keep pass A (Jev decision head + length head) and the masked-diffusion slot block. Add three
argument mechanisms that pick instead of spell: a copy/generate switch with a span pointer into the query,
Jev-style option slots for bools/enums, and per-item list slots. Separately: relabel training arguments with a
strong teacher (distillation), measure LLM escalation that reuses ActionMan's decisions, and try CUDA graphs.

**Tech Stack:** ModernBERT-large (HF transformers 5.19), PyTorch 2.14, Qwen3-0.6B baseline, Qwen3-32B-FP8 teacher
under vLLM, Vast.ai (RTX 4090 / H100), private HF repo `shing-dev/actionman`.

**Spec:** `docs/specs/2026-10-13-actionman-v1-design.md` (v1 design) + this plan's "Design" section (v3 deltas,
agreed in chat 2026-10-14). Results to beat: `docs/results/findings.md`.

## Global Constraints

- Spend: account $27.53 at plan time. **Hard stop for tonight: never launch anything that could take credit
  below $12** (`launch.py` floor check set to 12). Expected spend ~$8, caps total $15.
- Never read/print secrets; `HF_TOKEN` only via `infra/launch.py`'s existing path. Don't run env-sync here.
- Every box: L1 self-destroy (`box_job.sh`), L2 watchdog (10 min stale heartbeat), per-run `--cap-hours`.
  The laptop L3 watcher is re-registered at start and unregistered at the end.
- Eval data is never relabelled: BFCL gold + xLAM test stay as they are. Distillation touches training targets only.
- v2 behaviour must stay reproducible: all new behaviour sits behind `--v3` (stored in `run.json`), existing 32 tests
  keep passing.
- No emojis; findings written as measured numbers, with single-run noise (~3 pts on BFCL categories) stated.

## Review Focus

1. Value occurs several times in the query, or only case-differently ("boston" vs gold "Boston") -> copy target is
   the first token-aligned, case-insensitive occurrence; scorer is case-insensitive so this is correct.
2. Query text with non-ASCII characters (accents, CJK, emoji) -> copied text is sliced from the original query
   string by char offsets, never re-decoded from tokens, so it round-trips exactly.
3. Option param whose gold value is not among its options (label noise) -> falls back to a generate slot; no crash,
   no -100 leakage into the option loss.
4. List param with 0 items / more than 7 items / an item longer than 8 tokens -> absent / escalated / truncated
   (truncation counted in the audit, not silent).
5. Zero-parameter tools and option-only tools mixed with repeated calls (the v1 misalignment bug class) -> slot
   index stays aligned with call index.

---

## Budget and timeline

| Stage | What | GPU | Expected | Cap | Gate to continue |
|---|---|---|---|---|---|
| S0 | Data audit (copyable / options / lists / truncation) | laptop | $0 | $0 | decides which v3 parts are worth building |
| S1 | Build v3 (copy switch, option slots, list slots) + CPU tests | laptop | $0 | $0 | all tests green, tiny CPU train loss falls |
| S2 | GPU smoke: 300 steps v3 + eval 200 | RTX 4090 | $0.15 | $0.50 | no OOM, losses finite, eval runs |
| S3a | Distill pilot: Qwen3-32B-FP8 relabels 1,000 train examples | H100 | $0.50 | $1.00 | teacher changes >=3% of arg values AND full-run estimate <=2.5 h |
| S3b | Distill full relabel (~68k examples with calls) | H100 | $3.00 | $5.00 | acceptance rate >=70% |
| S4a | Train v3 on train_v2 (3 epochs, cached+markers) + eval T1-3 | RTX 4090 | $0.80 | $1.50 | - |
| S4b | Train v3 on distilled train_v2 + eval T1-3 | RTX 4090 | $0.80 | $1.50 | - |
| S5 | Escalation curve + latency bench (best v3 + Qwen v2) | RTX 4090 | $0.40 | $1.00 | - |
| S6 | CUDA graphs / torch.compile latency try (time-boxed 1 h) | RTX 4090 | $0.20 | $0.60 | - |
| S7 | Optional fairness run: Qwen3-0.6B on distilled data | RTX 4090 | $1.30 | $2.00 | only if S4b > S4a by >=2 pts and credit >= $16 |
| - | Failed boxes / retries reserve | - | $0.75 | $1.50 | - |
| | **Total** | | **~$8** | **$15.1** | credit after: ~$19.5 expected, never below $12 |

Order (parallel where independent): S0 -> S3a launched right after S0 (does not need v3 code) -> S1 while the
pilot/relabel runs -> S2 -> S4a and S4b in parallel -> S5 -> S6 -> S7 if gated in -> write-up.
Rough clock: S0 0:20, S1 3:00, S2 0:30, S4 2:00, S5 0:30, S6 1:00, write-up 0:45 -> ~8 h of wall time, fits before 11:00.

If a gate fails, the stage is skipped and the reason goes into findings; the rest continues.

## Design (v3 deltas, agreed 2026-10-14)

```
Pass B v3 slot block:
 #1 tool( city= [M][M][M][M] , unit= [M] ( [OPT] metric [OPT] imperial ) , tags= [ [M]x8 | [M]x8 ] )
          \_ value anchor = first [M]   \_ option anchor          \_ list: one 8-slot item per element

Per scalar value / list item (anchor state h):
  gate  = sigmoid(w . h)                         copy vs generate
  start = q_s(h) . k_s(h_query_tok)              span pointer over query tokens (like Jev over [T])
  end   = q_e(h) . k_e(h_query_tok)              best (s, e) with s <= e < s + 32
  copy  -> value = query[char(s).start : char(e).end]   (exact text; slots refilled with the span tokens)
  gen   -> masked diffusion as today
Per option param:  logits = q_o(h_anchor) . k_o(h_[OPT]_m)   (bool -> options [true, false]; enum <= 16)
Per list param:    pass-A count head on [h_call; h_arg; h_call*h_arg] -> {absent, 1..7, overflow}
```

Training adds four losses (gate BCE, span CE, option CE, count CE) to the existing three, all weight 1.

---

### Task 0: Data audit (S0)

**Files:**
- Create: `src/actionman/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Produces: `copy_span(tok, query: str, text: str) -> tuple[int, int] | None` (token indices within the query's own
  tokenization, inclusive) and `audit(tok, examples) -> dict` -> written to `docs/results/audit.md`.

- [ ] **Step 1: Write the failing test**

```python
from actionman.audit import copy_span


def test_copy_span_token_aligned_case_insensitive(tok):
    q = "Find sushi in boston, MA tonight"
    s, e = copy_span(tok, q, "Boston, MA")
    offs = tok(q, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    assert q[offs[s][0]: offs[e][1]].lower() == "boston, ma"


def test_copy_span_rejects_partial_token_and_absent(tok):
    assert copy_span(tok, "weather in NYC", "NY") is None       # not token-aligned
    assert copy_span(tok, "weather in NYC", "London") is None
```

- [ ] **Step 2: Run to verify it fails** — `.venv/Scripts/python.exe -m pytest tests/test_audit.py -q` -> ImportError.

- [ ] **Step 3: Implement**

```python
"""Data audit: how much of the argument mass can be copied / picked from options / is a list."""
from __future__ import annotations

from collections import Counter

from .schema import value_to_text


def copy_span(tok, query: str, text: str):
    """First case-insensitive, token-aligned occurrence of `text` in `query` -> (start_tok, end_tok) or None."""
    text = text.strip()
    if not text:
        return None
    offs = tok(query, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    starts = {a: i for i, (a, _) in enumerate(offs)}
    ends = {b: i for i, (_, b) in enumerate(offs)}
    low, t = query.lower(), text.lower()
    at = low.find(t)
    while at != -1:
        s = starts.get(at) if at in starts else starts.get(at - 1) if query[at - 1:at] == " " else None
        e = ends.get(at + len(t))
        if s is not None and e is not None and e >= s:
            return s, e
        at = low.find(t, at + 1)
    return None
```

(The `at - 1` branch covers BPE tokens that carry their leading space, e.g. `" Boston"` starts one char early.)

`audit(tok, examples)` counts, per source, over every gold argument: type (str/num/bool/list/dict/other),
`copyable` (str/num whose text has a `copy_span`), `has_options` (bool or captured enum), list item counts and
items longer than 8 tokens, values longer than 32 tokens. `__main__` runs it on `train_v2.jsonl`, `test.jsonl`,
`bfcl.jsonl` and writes `docs/results/audit.md`.

- [ ] **Step 4: Run tests** -> PASS. Run the audit: `PYTHONPATH=src .venv/Scripts/python.exe -m actionman.audit`.
- [ ] **Step 5: Apply the gates** and record them in the KB log:
  - copyable share of str values on BFCL < 30% -> drop the copy switch (Task 2), keep the rest.
  - option-typed params < 3% of BFCL args -> drop option slots (Task 3).
  - list args < 3% of BFCL args -> drop list slots (Task 4).
- [ ] **Step 6: Leave uncommitted.**

### Task 1: Schema and pass-A plumbing (options, query offsets, count labels)

**Files:**
- Modify: `src/actionman/schema.py` (`Param`, `from_bfcl`, `example_to/from_dict`), `src/actionman/extra.py`
  (`from_hermes`, `from_toolace` keep `enum`), `src/actionman/features.py` (`SPECIAL`, `build_pass_a`)
- Test: `tests/test_features.py`, `tests/test_schema.py`

**Interfaces:**
- Produces: `Param.options: list | None = None`; `param_options(p) -> list | None` (bool -> `[True, False]`,
  enum -> first 16, else None); `SPECIAL` gains `"[OPT]"` (embedding still pads to 50,432);
  `build_pass_a(..., v3=False)` adds `q_pos: list[int]` (sequence positions of query tokens), `q_offs` (their char
  offsets) and `count_items: list[(i, j, k, c)]` with c in 0..8 (0 absent, 8 overflow) for list params.
  `data/prepared/*.jsonl` must be re-prepared so enums survive (`prepare --plus`, `--v2`; ~10 min laptop).

- [ ] **Step 1: Failing tests**

```python
def test_param_options_bool_and_enum():
    from actionman.schema import Param, param_options
    assert param_options(Param("x", "bool")) == [True, False]
    assert param_options(Param("u", "str", options=["metric", "imperial"])) == ["metric", "imperial"]
    assert param_options(Param("n", "int")) is None


def test_pass_a_v3_query_positions_and_counts(tok, ex_list):
    a = build_pass_a(tok, ex_list, v3=True)
    q_text = tok.decode([a["input_ids"][p] for p in a["q_pos"]])
    assert q_text.strip() == ex_list.query.strip()
    assert (0, 0, 0, 2) in a["count_items"]          # ex_list: tags=["a", "b"] on param 0
```

(`ex_list` fixture added to `tests/conftest.py`: one tool `tag(tags: list)`, query `tag a and b`, call `tags=["a","b"]`.)

- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement** (offsets via `return_offsets_mapping=True` on the
  query only; `q_pos` = contiguous range after `[USER]`). - [ ] **Step 4: All tests pass (32 + new).**
- [ ] **Step 5: Leave uncommitted.**

### Task 2: Copy/generate switch (pass B + model + training targets)

**Files:**
- Modify: `src/actionman/features.py` (`build_pass_b(..., v3=False)`), `src/actionman/model.py`
  (heads + `loss_v3`), `src/actionman/data.py` (`featurize`, `collate`, `_collate_cached` carry v3 targets)
- Create: `src/actionman/span.py` (`best_span`)
- Test: `tests/test_v3.py`

**Interfaces:**
- Consumes: Task 0 `copy_span`, Task 1 `q_pos`, `q_offs`.
- Produces: pass-B dict gains `anchors: list[int]` (first slot position per value, same order as `slot_index`),
  `copy_targets: list[(s_pos, e_pos) | None]` (sequence positions); `ActionMan.copy_logits(h, anchor_b, anchor_p,
  q_b, q_p) -> (gate[N], start[N,Q], end[N,Q])`; `best_span(start, end, max_len=32) -> (s, e, prob)`.

- [ ] **Step 1: Failing tests**

```python
def test_best_span_respects_order_and_max_len():
    import torch
    from actionman.span import best_span
    st = torch.tensor([0.0, 5.0, 0.0, 0.0]); en = torch.tensor([9.0, 0.0, 0.0, 4.0])
    s, e, p = best_span(st, en, max_len=32)
    assert (s, e) == (1, 3) and 0 < p <= 1                   # end=0 is before start, so not allowed


def test_pass_b_v3_copy_target_points_at_query(tok, ex_copy):
    a = build_pass_a(tok, ex_copy, v3=True)
    b = build_pass_b(tok, ex_copy, a, gold_plan(a, ex_copy), with_targets=True, v3=True)
    s, e = b["copy_targets"][0]
    assert tok.decode(b["input_ids"][s: e + 1]).strip().lower() == "boston"


def test_v3_loss_learns_on_tiny_model(tiny_model, tok, ex_copy):
    # same pattern as tests/test_model.py: 30 steps on one batch, total loss must drop by >= 50%
    ...
```

(The third test reuses the existing tiny-model learning-test helper in `tests/test_model.py`; it asserts the new
`gate` and `span` loss terms are present in the returned dict and the total drops.)

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**

```python
# span.py
import torch


def best_span(start: torch.Tensor, end: torch.Tensor, max_len: int = 32):
    """Highest p(start)*p(end) with start <= end < start + max_len."""
    ps, pe = start.float().softmax(-1), end.float().softmax(-1)
    Q = ps.shape[0]
    i = torch.arange(Q)
    ok = (i[None, :] >= i[:, None]) & (i[None, :] < i[:, None] + max_len)
    score = (ps[:, None] * pe[None, :]).masked_fill(~ok, 0)
    flat = int(score.argmax())
    return flat // Q, flat % Q, float(score.flatten()[flat])
```

```python
# model.py additions (in __init__)
self.gate = nn.Linear(d, 1)
self.sq, self.sk, self.eq, self.ek = (nn.Linear(d, d) for _ in range(4))

def copy_logits(self, h, ab, ap, qb, qp):
    """ab/ap: anchor batch/pos [N]; qb/qp: query batch/pos [N,Q] padded with -1 (masked)."""
    ha = h[ab, ap]                                                  # [N,d]
    hq = h[qb.clamp(min=0), qp.clamp(min=0)]                        # [N,Q,d]
    m = qp < 0
    st = torch.einsum("nd,nqd->nq", self.sq(ha), self.sk(hq)) * self.scale
    en = torch.einsum("nd,nqd->nq", self.eq(ha), self.ek(hq)) * self.scale
    return self.gate(ha).squeeze(-1), st.masked_fill(m, float("-inf")), en.masked_fill(m, float("-inf"))
```

Loss: `BCEWithLogits(gate, copyable)` on all anchors; span CE on start/end only where copyable. In `_mask_slots`,
the anchor slot is always masked during training so the gate never sees its own answer.

- [ ] **Step 4: Tests pass.** - [ ] **Step 5: Leave uncommitted.**

### Task 3: Option slots (Jev-style pick for bools / enums)

**Files:**
- Modify: `src/actionman/features.py` (option block in `build_pass_b` when `v3`), `src/actionman/model.py`
  (`opt_q`, `opt_k`, `option_logits`), `src/actionman/data.py`
- Test: `tests/test_v3.py`

**Interfaces:**
- Consumes: Task 1 `param_options`, `[OPT]`.
- Produces: pass-B dict `opt_slots: list[(i, k, anchor_pos, [opt_pos...], target_idx | -100)]`;
  `ActionMan.option_logits(h, b, anchor_pos, opt_pos[N,M], opt_mask) -> [N,M]`.

- [ ] **Step 1: Failing tests**

```python
def test_option_block_layout_and_target(tok, ex_opt):          # unit in [metric, imperial], gold "imperial"
    a = build_pass_a(tok, ex_opt, v3=True)
    b = build_pass_b(tok, ex_opt, a, gold_plan(a, ex_opt), with_targets=True, v3=True)
    (i, k, anchor, opts, tgt), = b["opt_slots"]
    assert all(b["input_ids"][p] == tok.am["[OPT]"] for p in opts) and tgt == 1


def test_option_gold_not_in_options_falls_back_to_generate(tok, ex_opt_noisy):   # gold "kelvin"
    a = build_pass_a(tok, ex_opt_noisy, v3=True)
    b = build_pass_b(tok, ex_opt_noisy, a, gold_plan(a, ex_opt_noisy), with_targets=True, v3=True)
    assert b["opt_slots"] == [] and len(b["slot_pos"]) == 1
```

- [ ] **Step 2: fail.** - [ ] **Step 3: Implement.** Option param layout: `p= [MASK] ( [OPT] v1 [OPT] v2 ... )`;
  value text of each option via `value_to_text`; match gold with `scoring.value_match`. At inference, which params
  get an option block is known from the signature (no gold needed). - [ ] **Step 4: pass.** - [ ] **Step 5: uncommitted.**

### Task 4: List slots (count head + per-item slots)

**Files:**
- Modify: `src/actionman/features.py`, `src/actionman/model.py` (`count_head`), `src/actionman/data.py`,
  `src/actionman/infer.py`
- Test: `tests/test_v3.py`

**Interfaces:**
- Consumes: Task 1 `count_items`; Task 2 per-item anchors (each item is its own copy-or-generate value).
- Produces: `ITEM_SLOTS = 8`, `MAX_ITEMS = 7`; pass-B layout `p= [ [M]x8 | [M]x8 ]`; `slot_index` entries become
  `(i, k, item)` with `item = None` for scalars; `ActionMan.count_logits(h, idx) -> [N, 9]`.

- [ ] **Step 1: Failing tests**

```python
def test_list_items_get_own_slots_and_reassemble(tok, ex_list):
    a = build_pass_a(tok, ex_list, v3=True)
    b = build_pass_b(tok, ex_list, a, gold_plan(a, ex_list), with_targets=True, v3=True)
    assert [ix[2] for ix in b["slot_index"]] == [0, 1] and all(len(p) == 8 for p in b["slot_pos"])


def test_list_overflow_is_escalated_not_truncated(tok, ex_list_long):      # 9 items
    a = build_pass_a(tok, ex_list_long, v3=True)
    assert any(c == 8 for (*_, c) in a["count_items"])
```

- [ ] **Step 2: fail.** - [ ] **Step 3: Implement**; inference joins items into a Python list and converts each with
  `text_to_value` on the list's inner type (str when unknown). - [ ] **Step 4: pass.** - [ ] **Step 5: uncommitted.**

### Task 5: v3 inference, evaluate, train flag

**Files:**
- Modify: `src/actionman/infer.py` (`predict(..., v3=False)`), `src/actionman/evaluate.py` (`uses_v3`),
  `src/actionman/train.py` (`--v3`, written to `run.json`), `bench/latency.py`
- Test: `tests/test_infer.py`

**Interfaces:**
- Produces: `predict(model, tok, ex, T, tau, device, cached, markers, v3)` -> same `Prediction` type; `conf_args`
  for copied values = gate prob x span prob, for options = option prob, for lists = min over items.

Inference order per pass: (1) options -> argmax; (2) gate > 0.5 -> best_span, write span token ids into that value's
slots (cut/pad to the slot count), mark them unmasked with their confidence; (3) remaining masked slots ->
MLM as today; (4) re-mask below tau; stop early when nothing is below tau. Copied text = original query substring.

- [ ] **Step 1: Failing test**

```python
def test_predict_v3_end_to_end_well_formed(tiny_v3_model, tok, ex_mixed):     # str + option + list params
    p = predict(tiny_v3_model, tok, ex_mixed, T=2, v3=True)
    for c in p.calls:
        assert set(c.args) <= {pp.name for t in ex_mixed.tools for pp in t.params}
        assert all(0.0 <= v <= 1.0 for v in c.conf_args.values())
```

- [ ] **Step 2: fail.** - [ ] **Step 3: implement.** - [ ] **Step 4: full suite pass**
  (`.venv/Scripts/python.exe -m pytest -q`). Tiny CPU train run:
  `PYTHONPATH=src .venv/Scripts/python.exe -m actionman.train --tiny --v3 --cached --markers --max-steps 40 --out runs/dry3`
  -> loss falls, `DONE` written. - [ ] **Step 5: uncommitted.**

### Task 6: Distillation relabel (S3a pilot, S3b full)

**Files:**
- Create: `src/actionman/distill.py` (`teacher_prompt`, `merge_teacher`, `__main__` running vLLM on the box)
- Modify: `infra/launch.py` (`--image`, `--max-dph` already exists; floor check -> 12)
- Test: `tests/test_distill.py`

**Interfaces:**
- Produces: `merge_teacher(orig: list[Call], teacher: list[Call] | None, query: str) -> tuple[list[Call], dict]`;
  output file `train_v2_distilled.jsonl` + `distill_stats.json` (accepted calls %, changed args %, tokens/s).

Acceptance rule (keeps truth from the original label, takes surface form from the teacher):
- Teacher output must call the same multiset of tool names, else keep the original example unchanged.
- Per argument: take the teacher value if `scoring.value_match` says equal, or numbers are equal, or one normalised
  string contains the other ("American" / "American Airlines", "Boston" / "Boston, MA"); lists element-wise.
- Otherwise keep the original value. Teacher-added optional args kept only if their text is copyable from the query.

- [ ] **Step 1: Failing tests**

```python
from actionman.distill import merge_teacher
from actionman.schema import Call


def test_merge_takes_teacher_surface_form_keeps_original_truth():
    orig = [Call("f", {"airline": "American", "city": "Paris"})]
    teach = [Call("f", {"airline": "American Airlines", "city": "London"})]
    calls, st = merge_teacher(orig, teach, "fly American to Paris")
    assert calls[0].args == {"airline": "American Airlines", "city": "Paris"} and st["changed"] == 1


def test_merge_rejects_different_tools():
    calls, st = merge_teacher([Call("f", {"x": 1})], [Call("g", {"x": 1})], "q")
    assert calls[0].name == "f" and st["rejected_call"] == 1
```

- [ ] **Step 2: fail.** - [ ] **Step 3: implement** (`teacher_prompt` = Qwen chat template with the example's own
  tools and `enable_thinking=False`; vLLM greedy, `max_tokens=256`; parse with `baseline.qwen.parse_calls`).
- [ ] **Step 4: pass.**
- [ ] **Step 5: Pilot** (S3a):
  `infra/launch.py --run s8-distill-pilot --gpu "H100 SXM" --max-dph 2.6 --min-gpu-ram 80 --disk 80 --image vllm/vllm-openai:latest --cap-hours 0.6 --cmd 'python -m actionman.distill --model Qwen/Qwen3-32B-FP8 --in data/prepared/train_v2.jsonl --limit 1000 --out $OUT/pilot'`
  Gate: changed args >= 3% and projected full-run time <= 2.5 h. If time > 2.5 h, rerun the pilot with
  `Qwen/Qwen3-14B-FP8`; if the change rate < 3%, skip S3b/S4b (labels are already consistent; saves ~$4).
- [ ] **Step 6: Full** (S3b): same command without `--limit`, `--cap-hours 3`. Gate: accepted calls >= 70%.
- [ ] **Step 7: uncommitted.**

### Task 7: GPU runs S2 / S4a / S4b

- [ ] S2 smoke (cap 0.5 h, 4090): `python -m actionman.train --out $OUT/s --v3 --cached --markers --train-file train_v2.jsonl --max-steps 300 --tokens-per-batch 6144 --grad-ckpt && python -m actionman.evaluate --ckpt $OUT/s --T 2 --limit 200`
- [ ] S4a (cap 3 h): `python -m actionman.train --out $OUT/v3 --v3 --cached --markers --train-file train_v2.jsonl --epochs 3 --tokens-per-batch 6144 --grad-ckpt --workers 6 && python -m actionman.evaluate --ckpt $OUT/v3 --T 1 2 3`
- [ ] S4b (cap 3 h): same with `--train-file train_v2_distilled.jsonl --out $OUT/v3d`.
- [ ] Results into `infra/results.py` table; error files pulled and the five error classes from 2026-10-14
  (garbled, list soup, truncation, reasoning, copy-length) re-counted on v2 vs v3 for the same prompts.

### Task 8: Escalation curve (S5)

**Files:**
- Create: `bench/escalate.py`
- Test: `tests/test_escalate.py`

**Interfaces:**
- Consumes: `infer.predict` (v3), `baseline.qwen.load`, `baseline.qwen.parse_calls`, Qwen v2 checkpoint
  `runs/s5-qwen-v2/qwen` (verify on HF first; fall back to `s3-qwen` and say so).
- Produces: `assisted_prefix(pred, ex, theta) -> (tools_subset, prefix_text) | None` and `escalate.json`:
  for theta in {0.3, 0.5, 0.7, 0.85}: escalation rate, full-call accuracy, p50/mean/p95 latency, for three modes:
  `none` (ActionMan only), `full` (escalated prompt -> Qwen from scratch with all tools), `assisted` (Qwen sees only
  ActionMan's chosen tools; its answer is pre-filled with every confident call and the shaky call up to its first
  uncertain value, confident args first; Qwen only writes the rest).

- [ ] **Step 1: Failing tests**

```python
def test_assisted_prefix_prefills_confident_args_and_stops_at_shaky_one(ex_bmi, pred_bmi):
    # pred_bmi: weight=85 conf 1.0, height=60 conf 0.62, unit="metric" conf 0.99; theta 0.7
    tools, prefix = assisted_prefix(pred_bmi, ex_bmi, theta=0.7)
    assert [t.name for t in tools] == ["calculate_bmi"]
    assert prefix.endswith('"arguments": {"weight": 85, "unit": "metric", "height": ')


def test_no_escalation_when_all_confident(ex_bmi, pred_bmi_confident):
    assert assisted_prefix(pred_bmi_confident, ex_bmi, theta=0.7) is None
```

- [ ] **Step 2: fail.** - [ ] **Step 3: implement** (Hermes `<tool_call>` format, chat template with `tools=` the
  subset, assistant turn started with the prefix; parse prefix + continuation). - [ ] **Step 4: pass.**
- [ ] **Step 5: Run** (cap 1 h, 4090): `python bench/escalate.py --actionman $OUT/... --qwen <qwen ckpt> --n 2340`
  for accuracy, plus batch-1 latency on 300 prompts, same box.
- [ ] **Step 6: uncommitted.**

### Task 9: CUDA graphs try (S6, time-boxed 1 h)

- [ ] `bench/latency.py --compile`: `torch.compile(mode="reduce-overhead")` on the cached slot pass with the slot block
  padded to 64/128/256 and the context padded to multiples of 128 (static shapes). Report p50/mean vs eager on the
  same box. If it does not work inside the hour, stop and record why; no further spend.

### Task 10: Write-up

- [ ] `docs/results/findings.md` new section "v3" (same table layout: all, xLAM test, five BFCL categories, tool acc),
  ablation reading (v2 -> v3 = new heads; v3 -> v3d = distillation), error-class recount, escalation table,
  compile latency, costs via `infra/costs.py`. Note xLAM-test is single-answer gold, so a distilled model's style
  changes can cost points there that BFCL (multi-answer) does not penalise; report both.
- [ ] Regenerate `results.md`, `costs.md`; KB `log.md`, `tasks.md`, `decisions.md`, `00-overview.md`.
- [ ] Confirm zero running instances, unregister `ActionManWatcher`, final `pytest -q`.
- [ ] Hand-back message: headline, what changed, what failed, spend, what to verify. Nothing committed.
