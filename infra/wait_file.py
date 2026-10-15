"""On the box: wait (up to N minutes) until a file exists in the private HF repo. python infra/wait_file.py <path> [minutes]"""
import os
import sys
import time

from huggingface_hub import HfApi

path, minutes = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 40
api, t0 = HfApi(token=os.environ["HF_TOKEN"]), time.time()
while time.time() - t0 < minutes * 60:
    if path in api.list_repo_files(os.environ["HF_REPO"]):
        print("found", path, flush=True)
        sys.exit(0)
    open(os.environ.get("HEARTBEAT", "/tmp/hb"), "w").write(str(time.time()))
    time.sleep(30)
sys.exit(f"timed out waiting for {path}")
