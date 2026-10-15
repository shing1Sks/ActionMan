# ActionMan v4 plan: noise, copy boundaries, more data (2026-10-15)

Approved by the user: options 1-3 ("do 1-3, get more data"). No LLM hand-off (rejected: it breaks the one-small-model promise).
Nothing is committed; results are handed back for review.

## Questions

1. **Noise.** How much do single runs move? v3 (78.6) vs v3.1 (77.6) is undecided until we know.
2. **Copy boundaries.** Does widening copies cut mid-word ("US Doll" -> "US Dollars") help?
3. **More data / longer training.** Does 145k rows (train_v4) or 5 epochs beat 80k rows x 3 epochs?

## What the error audit changed (laptop, $0)

- Of v3's 85 "span length" errors, most are label conventions: we copy "Paris", the label says "Paris, France", which is not in the query.
- About 15-20 are real mid-word cuts. This is what option 2 targets now: an inference-only snap to whole words/numbers (`--snap`), so it needs no retraining.
- Only 5 errors involve numeric params, and all 5 are unit conversions (1.267 billion -> 1267000000). The planned number-only copy rule would fix none of them, so it was dropped.

## New data (train_v4 = train_v2 + 63,294 rows = 144,753)

| source | license | converted | kept | note |
|---|---|---|---|---|
| argilla/Synth-APIGen-v0.1 | Apache-2.0 | 45,361 | 42,983 (1,960 no-call) | xLAM-style; "lacks parameters" rows skipped |
| glaiveai/glaive-function-calling-v2 | Apache-2.0 | 51,084 | 6,311 (212 no-call) | only 6,316 unique queries; first turn only |
| Agent-Ark/Toucan-1.5M (1 of 3 SFT shards) | Apache-2.0 | 25,171 | 15,000 | real MCP tools; first step only, kept when it already makes every target call |
| nvidia/When2Call | CC-BY-4.0 | - | 0 | dropped: 5,162/5,191 queries are xLAM training queries with unusable tools (= our existing negatives) |

- Removed: queries matching dev/test/BFCL (16), duplicate queries, and queries <97% ASCII.
- Code: `extra.py` (converters), `prepare.py --v4`.

## Runs (RTX 4090, v3 recipe: --v3 --cached --markers, tokens 6144, grad-ckpt)

| run | change | eval | budget (expected / cap) |
|---|---|---|---|
| s15-v3-s1, s15-v3-s2 | `--v3-mask legacy --seed 1/2` (v3 as run on 10-14) | T1, T3, T3 --snap | $0.70 / $1.50 each |
| s15-v31-s1, s15-v31-s2 | `--seed 1/2` (v3.1 clean masking) | T1, T3, T3 --snap | $0.70 / $1.50 each |
| s16-v3-5ep | legacy, train_v2, 5 epochs | T1, T3, T3 --snap | $1.10 / $2.00 |
| s16-v3-v4data | legacy, train_v4, 3 epochs | T1, T3, T3 --snap | $1.50 / $2.70 |
| s16-snap-v3 | existing s10 v3 checkpoint, T3 --snap | eval only | $0.10 / $0.30 |

- **Total:** expected ~$5.5, worst case ~$11.5. Credit at start is $23.00 (launch floor $12, user floor $5).
- **Reading the results:** seed 0 of v3/v3.1 is 78.6 / 77.6. A data or epoch change counts as real only if it beats the v3 seed spread.

## Safety

The usual L1/L2/L3 safety layers apply: self-destroy, watchdog, and the laptop watcher (re-registered for this night, removed at the end).

## Outcome

- Seed noise ~1.5 pts; v3 and v3.1 are a tie (78.3 vs 78.5 mean).
- 5 epochs 79.7 (single run, only result outside noise); train_v4 78.7 (no gain); snap dropped.
- Spent $5.83. Details: `docs/results/findings.md` section 0.
