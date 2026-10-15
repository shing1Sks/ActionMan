#!/bin/bash
# b1-bfcl box job (run by infra/box_job.sh from /workspace/am): official BFCL v4 single-turn harness.
# Gate (Qwen3-0.6B FC vs leaderboard; stop on fail) -> ActionMan (flat + last-user) -> own-eval strict rescore
# -> our Qwen SFT -> public baselines -> baselines on our own 2,340 prompts. Outputs land in $OUT (synced to HF).
set -u
(while true; do date +%s > "$HEARTBEAT"; sleep 60; done) &
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 BFCL_PROJECT_ROOT=$OUT/bfcl HF_HUB_DISABLE_PROGRESS_BARS=1
export PATH=$PATH:/workspace/v/bin   # bfcl spawns `vllm serve`; appended so `python` stays the ActionMan env
mkdir -p "$OUT/bfcl"
V=/workspace/v/bin/python
RUN="$V bench/bfcl_run.py"
CATS=single_turn
stage() { echo "== $* $(date -u +%H:%M:%S)"; }

stage install
python -m pip install -q --break-system-packages uv > "$OUT/install.log" 2>&1
python -m uv venv /workspace/v --python 3.12 >> "$OUT/install.log" 2>&1
python -m uv pip install --python $V "bfcl-eval==2026.3.23" "vllm==0.11.0" "transformers<5" soundfile >> "$OUT/install.log" 2>&1 || exit 1
$V -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" > "$OUT/versions.txt" 2>&1
python -c "import transformers, torch; print('actionman env: transformers', transformers.__version__, 'torch', torch.__version__)" >> "$OUT/versions.txt"
python infra/fetch_ckpt.py runs/s16-v3-5epb/m:/workspace/ckpt/am runs/s5-qwen-v2/qwen:/workspace/ckpt/qwen || exit 1
$V bench/sft_tf4.py /workspace/ckpt/qwen /workspace/ckpt/qwen4 || exit 1

gen() { $RUN cli generate --model "$1" --test-category $CATS --backend vllm --result-dir result "${@:2}" > "$OUT/gen_${1//\//_}.log" 2>&1; }
evl() { $RUN cli evaluate --model "$1" --test-category "${2:-$CATS}" --result-dir result --score-dir score ${3:-} > "$OUT/eval_${1//\//_}.log" 2>&1; }

stage gate   # b1-bfcl2: Qwen3-0.6B FC (thinking, temp 0.001) came out +1.4 NL / +2.5 Live vs the leaderboard with
             # per-category moves in both directions -> anchor the gate on the two non-thinking (near-deterministic)
             # baselines; Qwen3-0.6B FC runs again here to measure its run-to-run spread
for m in MadeAgents/Hammer2.1-0.5b ibm-granite/granite-4.0-350m Qwen/Qwen3-0.6B-FC; do stage "gate $m"; gen $m; evl $m; done
$RUN gate --score-dir "$OUT/bfcl/score" --models Qwen/Qwen3-0.6B-FC | tee "$OUT/gate_qwen.json"
$RUN gate --score-dir "$OUT/bfcl/score" --models MadeAgents/Hammer2.1-0.5b,ibm-granite/granite-4.0-350m | tee "$OUT/gate.json"
if [ "${PIPESTATUS[0]}" != "0" ]; then echo GATE_FAIL | tee "$OUT/GATE_FAIL"; exit 1; fi

stage actionman
for m in flat last; do
  python bench/bfcl_actionman.py --ckpt /workspace/ckpt/am --entries data/bench/bfcl_v4_entries.jsonl --mode $m \
    --out "$OUT/am_$m.jsonl" > "$OUT/am_$m.log" 2>&1
done
$RUN import --preds "$OUT/am_flat.jsonl" --model actionman-s16 | tee "$OUT/am_flat_status.json"
$RUN import --preds "$OUT/am_last.jsonl" --model actionman-s16-last | tee "$OUT/am_last_status.json"
evl actionman-s16
evl actionman-s16-last

stage own-eval-actionman
python -m actionman.evaluate --ckpt /workspace/ckpt/am --T 3 --dump --out "$OUT/am_own.json" > "$OUT/am_own.log" 2>&1
$RUN strict --preds "$OUT/am_own_preds.jsonl" --out "$OUT/am_own_strict.json" | tail -2

stage baselines
gen am-qwen-sft-FC --local-model-path /workspace/ckpt/qwen4; evl am-qwen-sft-FC
stage Qwen3-1.7B; gen Qwen/Qwen3-1.7B-FC; evl Qwen/Qwen3-1.7B-FC

stage own-eval-baselines
$RUN own --model am-qwen-sft-FC --local-model-path /workspace/ckpt/qwen4 --out "$OUT/own_am-qwen-sft-FC.json" > "$OUT/own_sft.log" 2>&1
for m in Qwen/Qwen3-0.6B-FC ibm-granite/granite-4.0-350m MadeAgents/Hammer2.1-0.5b Qwen/Qwen3-1.7B-FC; do
  stage "own $m"; $RUN own --model $m --out "$OUT/own_${m//\//_}.json" > "$OUT/own_${m//\//_}.log" 2>&1
done

stage final-evaluate
$RUN cli evaluate --model Qwen/Qwen3-0.6B-FC,actionman-s16,actionman-s16-last,am-qwen-sft-FC,ibm-granite/granite-4.0-350m,MadeAgents/Hammer2.1-0.5b,Qwen/Qwen3-1.7B-FC \
  --test-category $CATS --result-dir result --score-dir score_all > "$OUT/eval_all.log" 2>&1
stage done
