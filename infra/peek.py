"""Peek at running jobs over SSH: python infra/peek.py [n_lines]  (prints status + last log lines per ledger instance)."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import vast  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ledger = {json.loads(l)["instance_id"]: json.loads(l)["run"] for l in open(os.path.join(ROOT, "infra", "ledger.jsonl")) if l.strip()}
for i in vast.instances():
    iid = int(i["id"])
    print(f"== {iid} {ledger.get(iid, '?')} {i.get('gpu_name')} status={i.get('actual_status')} gpu_util={i.get('gpu_util')}")
    host, port = i.get("public_ipaddr"), (i.get("ports") or {}).get("22/tcp", [{}])[0].get("HostPort")
    if not (host and port):
        continue
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR", "-p", str(port), f"root@{host}",
           f"grep -v -i -E 'warn|it/s|%\\|' /workspace/out/job.log 2>/dev/null | tail -{n}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=40).stdout
        print("\n".join("   " + l[:240] for l in out.splitlines() if "Welcome" not in l and "Have fun" not in l))
    except subprocess.TimeoutExpired:
        print("   (ssh timeout)")
