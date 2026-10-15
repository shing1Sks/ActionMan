"""Poll one instance until it is gone (self-destroyed); print status changes and new job.log lines from HF.
python infra/monitor.py <instance_id> <run_id> [poll_seconds]"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import vast  # noqa: E402
from launch import REPO, hf_token  # noqa: E402

iid, run = int(sys.argv[1]), sys.argv[2]
poll = int(sys.argv[3]) if len(sys.argv) > 3 else 60
token, last_status, seen = hf_token(), None, 0
t0 = time.time()
while True:
    try:
        inst = [i for i in vast.instances() if int(i["id"]) == iid]
    except Exception as e:
        print("poll error", e, flush=True); time.sleep(poll); continue
    status = inst[0].get("actual_status") if inst else "GONE"
    if status != last_status:
        print(f"[{(time.time() - t0) / 60:5.1f}m] status {status}", flush=True); last_status = status
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, f"runs/{run}/job.log", token=token, force_download=True)
        lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        for line in lines[seen:]:
            if "Warning" not in line and "warn(" not in line:
                print("  " + line[:300], flush=True)
        seen = len(lines)
    except Exception:
        pass
    if status == "GONE":
        print(f"instance gone after {(time.time() - t0) / 60:.1f} min; credit ${vast.credit():.2f}", flush=True)
        break
    time.sleep(poll)
