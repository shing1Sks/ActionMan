#!/bin/bash
# Vast onstart: fetch bundle from the private HF repo and start the self-destructing job in the background.
# If the bundle can't be fetched (bad host network), destroy this instance immediately instead of idling.
env | grep -E '^(CONTAINER_|RUN_ID|HF_REPO|CAP_SECONDS|JOB_CMD_B64|HF_TOKEN|VAST_)' >> /etc/environment
mkdir -p /workspace/am && cd /workspace/am
python -m pip install -q --break-system-packages huggingface_hub > /workspace/setup.log 2>&1
timeout 900 python - <<'EOF' >> /workspace/setup.log 2>&1
import os, tarfile
from huggingface_hub import hf_hub_download
for f in [f"bundles/{os.environ['RUN_ID']}/code.tar.gz", "bundles/data.tar.gz"]:
    p = hf_hub_download(os.environ["HF_REPO"], f, token=os.environ["HF_TOKEN"])
    tarfile.open(p).extractall("/workspace/am")
print("bundle ok")
EOF
if [ ! -f /workspace/am/infra/box_job.sh ]; then
  curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" -H "Authorization: Bearer $CONTAINER_API_KEY" >> /workspace/setup.log 2>&1
  exit 1
fi
nohup bash /workspace/am/infra/box_job.sh > /workspace/job_outer.log 2>&1 &
