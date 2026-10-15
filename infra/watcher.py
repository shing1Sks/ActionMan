"""L3 (laptop): every run destroys instances that are over 1.5x their cap, not in the ledger, or stopped;
destroys everything if credit < $5. Run by the ActionManWatcher scheduled task. --dry-run only reports."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import vast  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LEDGER = os.path.join(ROOT, "infra", "ledger.jsonl")
LOG = os.path.join(ROOT, "infra", "watcher.log")
FLOOR, SLACK = 5.0, 1.5  # user hard floor: destroy everything below $5


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def main(dry: bool):
    ledger = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            if line.strip():
                r = json.loads(line)
                ledger[int(r["instance_id"])] = r
    try:
        insts, credit = vast.instances(), vast.credit()
    except Exception as e:
        log(f"ERROR polling vast: {e}")
        return
    kill = []
    for i in insts:
        iid, state = int(i["id"]), (i.get("actual_status") or i.get("cur_state") or "?")
        r = ledger.get(iid)
        if credit < FLOOR:
            kill.append((iid, f"credit ${credit:.2f} below floor"))
        elif r is None:
            # grace: launch.py writes the ledger only after `vastai create` returns, so a brand-new box can be
            # visible here for a minute or two before its ledger line exists
            if time.time() - float(i.get("start_date") or 0) > 600:
                kill.append((iid, f"not in ledger (label={i.get('label')})"))
        elif state in ("exited", "stopped", "offline") or i.get("cur_state") == "stopped":
            kill.append((iid, f"state {state}"))
        elif time.time() - r["started_at"] > SLACK * r["cap_hours"] * 3600:
            kill.append((iid, f"over {SLACK}x cap ({r['cap_hours']}h)"))
    for iid, why in kill:
        if dry:
            print(f"would destroy {iid}: {why}")
            continue
        try:
            vast.destroy(iid)
            log(f"DESTROYED {iid}: {why}")
        except Exception as e:
            log(f"ERROR destroying {iid}: {e}")
    log(f"ok credit=${credit:.2f} instances={[(int(i['id']), i.get('actual_status')) for i in insts]} actions={len(kill)}"
        + (" (dry)" if dry else ""))
    if dry:
        print(f"credit ${credit:.2f}; instances {len(insts)}; actions {len(kill)}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
