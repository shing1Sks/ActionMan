#!/bin/bash
# L1: run JOB_CMD under a hard timeout; on ANY exit upload outputs to the private HF repo, then destroy this instance.
set -u
cd /workspace/am
export PYTHONPATH=/workspace/am/src PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
export OUT=/workspace/out HEARTBEAT=/workspace/out/heartbeat PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
date +%s > "$HEARTBEAT"

upload() {  # $1 = "final" uploads everything incl. weights; otherwise logs/metrics only
  python infra/hf_sync.py "$OUT" "runs/$RUN_ID" "${1:-logs}" >> /workspace/sync.log 2>&1
}
destroy() {
  echo "destroying $CONTAINER_ID at $(date)" >> "$OUT/job.log"
  upload logs
  # Vast REST API with the instance-scoped key (no CLI dependency)
  # retried: a host with flaky DNS once failed this single call and kept billing until the laptop watcher
  for i in 1 2 3 4 5 6 7 8 9 10; do
    curl -sf -X DELETE "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" -H "Authorization: Bearer $CONTAINER_API_KEY" >> /workspace/sync.log 2>&1 && break
    sleep 30
  done
}
finish() { upload final; destroy; }
trap finish EXIT

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt" 2>&1
cp /workspace/setup.log "$OUT/setup.log" 2>/dev/null
python -m pip install -q --break-system-packages "transformers==5.19.0" accelerate safetensors >> "$OUT/setup.log" 2>&1
echo "setup done $(date)" >> "$OUT/job.log"

python infra/watchdog.py "$HEARTBEAT" 600 "$$" >> "$OUT/watchdog.log" 2>&1 &
( while true; do sleep 300; upload logs; done ) &

CMD="$(echo "$JOB_CMD_B64" | base64 -d)"
echo "cmd: $CMD" >> "$OUT/job.log"
timeout "$CAP_SECONDS" bash -c "$CMD" >> "$OUT/job.log" 2>&1
echo "exit code $? at $(date)" >> "$OUT/job.log"
