"""Upload a run's output folder to the private HF repo. mode=logs skips weights."""
import os
import sys

from huggingface_hub import HfApi

folder, dest, mode = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "logs")
ignore = ["*.pt", "*.safetensors", "*.bin", "heartbeat"] if mode != "final" else ["heartbeat"]
import time
for attempt in range(4):
    try:
        HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
            folder_path=folder, path_in_repo=dest, repo_id=os.environ["HF_REPO"], ignore_patterns=ignore,
            commit_message=f"{dest} ({mode})")
        print("synced", dest, mode, flush=True)
        break
    except Exception as e:  # network blips: retry before the box is destroyed
        print("sync failed", attempt, repr(e)[:300], flush=True)
        time.sleep(20 * (attempt + 1))
