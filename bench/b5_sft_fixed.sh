#!/bin/bash
# b5 box job: our Qwen SFT re-measured with a config transformers 4.x reads correctly (rope_theta 1e6; see sft_tf4.py).
# BFCL v4 single-turn + own eval (bf16, like every other model) and vLLM accuracy/latency on the 500 latency ids.
# (b4 was the same job measuring fp32/fp16; it showed precision was not the problem, the rope config was.)
set -u
(while true; do date +%s > "$HEARTBEAT"; sleep 60; done) &
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 BFCL_PROJECT_ROOT=$OUT/bfcl HF_HUB_DISABLE_PROGRESS_BARS=1
export PATH=$PATH:/workspace/v/bin PYTHONPATH=/workspace/am/src:/workspace/am
V=/workspace/v/bin/python
RUN="$V bench/bfcl_run.py"
L=$OUT/lat; mkdir -p "$OUT/bfcl" "$L"
stage() { echo "== $* $(date -u +%H:%M:%S)"; }

stage install
python -m pip install -q --break-system-packages uv > "$OUT/install.log" 2>&1
python -m uv venv /workspace/v --python 3.12 >> "$OUT/install.log" 2>&1
python -m uv pip install --python $V "bfcl-eval==2026.3.23" "vllm==0.11.0" "transformers<5" soundfile >> "$OUT/install.log" 2>&1 || exit 1
python infra/fetch_ckpt.py runs/s5-qwen-v2/qwen:/workspace/ckpt/qwen || exit 1
$V bench/sft_tf4.py /workspace/ckpt/qwen /workspace/ckpt/qwen4 || exit 1

stage latency
$V bench/latency.py --ids bench/latency_ids.json --n 500 --arms vllm --llm /workspace/ckpt/qwen4 \
  --label vllm_qwen_sft --out "$L/vllm_sft_fixed.json" > "$L/vllm_sft_fixed.log" 2>&1

stage bfcl
$RUN cli generate --model am-qwen-sft-FC --test-category single_turn --backend vllm --result-dir result \
  --local-model-path /workspace/ckpt/qwen4 > "$OUT/gen_sft.log" 2>&1
$RUN cli evaluate --model am-qwen-sft-FC --test-category single_turn --result-dir result --score-dir score > "$OUT/eval_sft.log" 2>&1
$RUN own --model am-qwen-sft-FC --local-model-path /workspace/ckpt/qwen4 --out "$OUT/own_am-qwen-sft-FC.json" > "$OUT/own_sft.log" 2>&1
stage done
