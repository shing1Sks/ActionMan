"""L2 (on the box): if the heartbeat file is stale for `limit` seconds, kill the job (box_job's trap uploads + destroys).
If that somehow fails, upload logs and destroy directly 5 minutes later."""
import os
import subprocess
import sys
import time

hb, limit, job_pid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fired_at = None
while True:
    time.sleep(30)
    try:
        age = time.time() - os.path.getmtime(hb)
    except OSError:
        age = limit + 1
    if age > limit and fired_at is None:
        print(f"{time.ctime()} heartbeat stale {age:.0f}s > {limit}s: killing job", flush=True)
        subprocess.run(["pkill", "-TERM", "-f", "timeout"], check=False)
        fired_at = time.time()
    if fired_at and time.time() - fired_at > 300:
        print(f"{time.ctime()} job still alive after kill: destroying directly", flush=True)
        subprocess.run(["python", "infra/hf_sync.py", "/workspace/out", f"runs/{os.environ['RUN_ID']}", "final"], check=False)
        subprocess.run(["curl", "-s", "-X", "DELETE", f"https://console.vast.ai/api/v0/instances/{os.environ['CONTAINER_ID']}/",
                        "-H", f"Authorization: Bearer {os.environ['CONTAINER_API_KEY']}"], check=False)
        break
