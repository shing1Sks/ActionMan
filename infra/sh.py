"""Run a shell command on a running box over SSH: python infra/sh.py <instance_id> '<cmd>'"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import vast  # noqa: E402

iid, cmd = int(sys.argv[1]), sys.argv[2]
i = next(x for x in vast.instances() if int(x["id"]) == iid)
host, port = i.get("public_ipaddr"), (i.get("ports") or {}).get("22/tcp", [{}])[0].get("HostPort")
r = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR", "-p", str(port), f"root@{host}", cmd],
                   capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
print(r.stdout[-4000:], r.stderr[-1000:])
