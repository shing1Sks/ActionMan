"""Per-run cost estimate: ledger start -> 'destroying ... at <UTC date>' line in the run's job.log (or now if alive)."""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import vast  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from launch import LEDGER, REPO, hf_token  # noqa: E402

alive = {int(i["id"]) for i in vast.instances()}
rows, total = [], 0.0
for line in open(LEDGER):
    if not line.strip():
        continue
    r = json.loads(line)
    end, how = None, "?"
    if r["instance_id"] in alive:
        end, how = time.time(), "running"
    else:
        try:
            log = open(hf_hub_download(REPO, f"runs/{r['run']}/job.log", token=hf_token(), force_download=True),
                       encoding="utf-8", errors="replace").read()
            m = re.findall(r"destroying \d+ at (.+)", log)
            if m:
                end = datetime.strptime(m[-1].strip(), "%a %b %d %H:%M:%S %Z %Y").replace(tzinfo=timezone.utc).timestamp()
                how = "self-destroyed"
        except Exception:
            pass
        if end is None:
            end, how = r["started_at"] + 0.5 * 3600, "no log (est. <=30 min)"
    hours = max(0.0, (end - r["started_at"]) / 3600)
    cost = hours * r["dph"]
    total += cost
    rows.append(f"| {r['run']} | {r['instance_id']} | {r['gpu']} | ${r['dph']:.3f}/h | {hours:.2f} h | ${cost:.2f} | {how} |")
print("| run | instance | GPU | rate | hours | cost | end |\n|---|---|---|---|---|---|---|")
print("\n".join(rows))
print(f"\nEstimated GPU total: ${total:.2f}; account credit now ${vast.credit():.2f} (started at $32.96)")
