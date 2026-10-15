#!/bin/bash
# b2-latency box job (run by infra/box_job.sh from /workspace/am): one RTX 4090, batch 1, the 500 frozen BFCL v4 ids.
# GPU arms twice in ABAB order (ActionMan eager+compiled, vLLM Qwen SFT x2 configs, vLLM Qwen3-0.6B public,
# llama.cpp Q8_0), HF generate once as a reference, then box-CPU arms pinned to 8 cores (ActionMan fp32 vs llama.cpp).
set -u
(while true; do date +%s > "$HEARTBEAT"; sleep 60; done) &
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=/workspace/am/src:/workspace/am
V=/workspace/v/bin/python
L=$OUT/lat; mkdir -p "$L" /workspace/llama /workspace/gguf
LB=b11623
IDS="--ids bench/latency_ids.json --n 500"
stage() { echo "== $* $(date -u +%H:%M:%S)"; }

stage install
(nvidia-smi; lscpu; nproc; free -g) > "$L/host.txt" 2>&1
python -m pip install -q --break-system-packages uv > "$OUT/install.log" 2>&1
python -m uv venv /workspace/v --python 3.12 >> "$OUT/install.log" 2>&1
python -m uv pip install --python $V "vllm==0.11.0" "transformers<5" >> "$OUT/install.log" 2>&1 || exit 1   # cu128 build (latest vLLM needs a CUDA 13 driver)
python infra/fetch_ckpt.py runs/s16-v3-5epb/m:/workspace/ckpt/am runs/s5-qwen-v2/qwen:/workspace/ckpt/qwen || exit 1
$V -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B')" >> "$OUT/install.log" 2>&1
$V bench/sft_tf4.py /workspace/ckpt/qwen /workspace/ckpt/qwen4 || exit 1   # SFT tokenizer files loadable by transformers 4.x
Q=/workspace/ckpt/qwen4

stage llama.cpp   # prebuilt release binaries + GGUF Q8_0 of our SFT; dropped if this takes > 20 min
timeout 1200 bash -c "
  cd /workspace/llama && mkdir -p gpu &&
  curl -sfL https://github.com/ggml-org/llama.cpp/releases/download/$LB/llama-$LB-bin-ubuntu-cuda-12.8-x64.tar.gz | tar xz -C gpu &&
  curl -sfL https://github.com/ggml-org/llama.cpp/releases/download/$LB/cudart-llama-$LB-bin-ubuntu-cuda-12.8-x64.tar.gz | tar xz -C gpu &&
  mkdir -p cpu && curl -sfL https://github.com/ggml-org/llama.cpp/releases/download/$LB/llama-$LB-bin-ubuntu-x64.tar.gz | tar xz -C cpu &&
  curl -sfL https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LB.tar.gz | tar xz &&
  python -m uv pip install --python $V ./llama.cpp-$LB/gguf-py sentencepiece &&
  $V llama.cpp-$LB/convert_hf_to_gguf.py /workspace/ckpt/qwen4 --outtype q8_0 --outfile /workspace/gguf/qwen-sft-q8_0.gguf
" > "$OUT/llama_setup.log" 2>&1
GPU_SRV=$(find /workspace/llama/gpu -name llama-server -type f | head -1)
CPU_SRV=$(find /workspace/llama/cpu -name llama-server -type f | head -1)
echo "llama gpu=$GPU_SRV cpu=$CPU_SRV gguf=$(ls -la /workspace/gguf)" >> "$OUT/llama_setup.log"

serve() {   # $1 server binary, rest = flags, optional $PIN prefix (taskset); waits for /health
  local bin=$1; shift
  LD_LIBRARY_PATH="$(dirname "$bin"):$(find "$(dirname "$bin")/.." -name 'libcudart.so*' -printf '%h' -quit 2>/dev/null):${LD_LIBRARY_PATH:-}" ${PIN:-} "$bin" "$@" --port 8080 -m /workspace/gguf/qwen-sft-q8_0.gguf -c 8192 -np 1 \
    > "$L/llama_server.log" 2>&1 &
  SRV=$!
  for i in $(seq 120); do curl -sf localhost:8080/health > /dev/null && return 0; sleep 2; done
  kill $SRV; return 1
}

for R in A B; do
  stage "round $R"
  python bench/latency.py $IDS --arms am_eager,am_compiled --actionman /workspace/ckpt/am --rep $R --out "$L/am_$R.json" > "$L/am_$R.log" 2>&1
  $V bench/latency.py $IDS --arms vllm --llm $Q --label vllm_qwen_sft --rep $R --out "$L/vllm_sft_$R.json" > "$L/vllm_sft_$R.log" 2>&1
  $V bench/latency.py $IDS --arms vllm --llm $Q --label vllm_qwen_sft_seqs1 --max-num-seqs 1 --rep $R --out "$L/vllm_sft1_$R.json" > "$L/vllm_sft1_$R.log" 2>&1
  $V bench/latency.py $IDS --arms vllm --llm Qwen/Qwen3-0.6B --label vllm_qwen3_0.6b_public --rep $R --out "$L/vllm_pub_$R.json" > "$L/vllm_pub_$R.log" 2>&1
  if [ -n "$GPU_SRV" ] && [ -f /workspace/gguf/qwen-sft-q8_0.gguf ] && serve "$GPU_SRV" -ngl 99; then
    $V bench/latency.py $IDS --arms llamacpp --llm $Q --label llamacpp_q8_gpu --rep $R --out "$L/llama_gpu_$R.json" > "$L/llama_gpu_$R.log" 2>&1
    kill $SRV; sleep 3
  fi
done

stage hf-reference
$V bench/latency.py $IDS --n 120 --arms hf --llm $Q --label hf_generate_qwen_sft --out "$L/hf_sft.json" > "$L/hf_sft.log" 2>&1

stage box-cpu   # same 8 cores and thread count for both engines, 200 prompts (SKIP_CPU=1: GPU-only rerun)
for R in $([ "${SKIP_CPU:-0}" = 1 ] || echo A B); do
  taskset -c 0-7 python bench/latency.py $IDS --n 200 --cpu --threads 8 --T 3 --arms am_eager --actionman /workspace/ckpt/am \
    --rep $R --out "$L/cpu_am_$R.json" > "$L/cpu_am_$R.log" 2>&1
  if [ -n "$CPU_SRV" ] && [ -f /workspace/gguf/qwen-sft-q8_0.gguf ] && PIN="taskset -c 0-7" serve "$CPU_SRV" -t 8 -ngl 0; then
    taskset -c 0-7 $V bench/latency.py $IDS --n 200 --threads 8 --arms llamacpp --llm $Q --label llamacpp_q8_cpu8 \
      --rep $R --out "$L/cpu_llama_$R.json" > "$L/cpu_llama_$R.log" 2>&1
    kill $SRV; sleep 3
  fi
done
cp /workspace/gguf/qwen-sft-q8_0.gguf "$OUT/" 2>/dev/null
stage done
