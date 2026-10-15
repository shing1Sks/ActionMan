r"""Bundle code+data to the private HF repo, rent the cheapest matching GPU, start a self-destructing job.

python infra/launch.py --run v1 --gpu "RTX 4090,A100 SXM4" --cap-hours 3 \
    --cmd "python -m actionman.train --out \$OUT/v1 --epochs 3"
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(__file__))
import vast  # noqa: E402
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = "shing-dev/actionman"
IMAGE = "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"
LEDGER = os.path.join(ROOT, "infra", "ledger.jsonl")
FLOOR = 8.0  # 2026-10-15: user approved spending existing credit; their hard floor is $5 (watcher)


def hf_token() -> str:
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8-sig"):
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("HF_TOKEN missing from .env")


def bundle(include_data: bool, extra_files: tuple = ()) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ["src", "baseline", "bench", "infra", "arena"]:
            tar.add(os.path.join(ROOT, d), arcname=d,
                    filter=lambda ti: None if ("__pycache__" in ti.name or ti.name.endswith((".jsonl", ".log", "replays.js"))
                                               or "arena/replays" in ti.name) else ti)
        if include_data:
            tar.add(os.path.join(ROOT, "data", "prepared"), arcname="data/prepared")
        for f in extra_files:   # e.g. data/prepared/train_plus.jsonl (shipped with this run's code bundle)
            tar.add(os.path.join(ROOT, f), arcname=f)
    return buf.getvalue()


def upload_bundle(token: str, run: str, extra_files: tuple = (), refresh_data: bool = False) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(REPO, private=True, exist_ok=True)
    files = {f"bundles/{run}/code.tar.gz": bundle(False, extra_files)}
    existing = set(api.list_repo_files(REPO))
    if refresh_data or "bundles/data.tar.gz" not in existing:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(os.path.join(ROOT, "data", "prepared"), arcname="data/prepared")
        files["bundles/data.tar.gz"] = buf.getvalue()
    for path, blob in files.items():
        api.upload_file(path_or_fileobj=blob, path_in_repo=path, repo_id=REPO)
        print(f"uploaded {path} ({len(blob) / 1e6:.1f} MB)")


def pick_offers(gpus: list[str], max_dph: float, min_gpu_ram: int, interruptible: bool, disk: int = 40) -> list[dict]:
    best = []
    for g in gpus:
        q = (f'gpu_name="{g}" num_gpus=1 reliability>0.98 disk_space>={disk} cuda_vers>=12.8 inet_down>=300 '
             f'direct_port_count>=1 gpu_ram>={min_gpu_ram} dph<={max_dph}')
        best += vast.offers(q, "dph", interruptible)
    bad = set()
    if os.path.exists(os.path.join(ROOT, "infra", "bad_machines.txt")):
        bad = {int(x) for x in open(os.path.join(ROOT, "infra", "bad_machines.txt")).read().split() if x.strip()}
    best = [o for o in best if o.get("inet_down_cost", 0) <= 0.02 and int(o.get("machine_id", -1)) not in bad]
    if not best:
        raise SystemExit("no matching offers")
    # value score: dph per unit of compute, mildly prefer reliability
    best.sort(key=lambda o: o["dph_total"] / max(o.get("total_flops", 1), 1) / max(o.get("reliability2", o.get("reliability", 0.98)), 0.9))
    return best[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", default="RTX 3090")
    ap.add_argument("--max-dph", type=float, default=0.6)
    ap.add_argument("--min-gpu-ram", type=int, default=22, help="GB")
    ap.add_argument("--cap-hours", type=float, required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--interruptible", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra-file", action="append", default=[])
    ap.add_argument("--refresh-data", action="store_true", help="re-upload bundles/data.tar.gz (data/prepared changed)")
    args = ap.parse_args()

    bal = vast.credit()
    gpus = [g.strip() for g in args.gpu.split(",")]
    offer = pick_offers(gpus, args.max_dph, args.min_gpu_ram, args.interruptible, args.disk)[0]
    est = offer["dph_total"] * args.cap_hours
    print(f"balance ${bal:.2f}; offer {offer['id']} {offer['gpu_name']} {offer.get('gpu_ram')}MB "
          f"${offer['dph_total']:.3f}/h {offer.get('geolocation')} rel {offer.get('reliability2', 0):.3f}; worst case ${est:.2f}")
    if bal - est < FLOOR:
        raise SystemExit(f"refusing: worst case would cross the ${FLOOR} floor")
    env = {"RUN_ID": args.run, "HF_REPO": REPO, "CAP_SECONDS": int(args.cap_hours * 3600),
           "JOB_CMD_B64": base64.b64encode(args.cmd.encode()).decode()}
    if args.dry_run:
        print("dry run: would create with env", {**env, "HF_TOKEN": "<redacted>"})
        return
    token = hf_token()
    upload_bundle(token, args.run, tuple(args.extra_file), args.refresh_data)
    onstart = os.path.join(ROOT, "infra", "onstart.sh")
    iid = None
    for offer in pick_offers(gpus, args.max_dph, args.min_gpu_ram, args.interruptible, args.disk):  # fresh list after upload
        try:
            iid = vast.create(offer["id"], IMAGE, args.disk, {**env, "HF_TOKEN": token}, onstart, f"am-{args.run}")
            break
        except RuntimeError as e:
            print(f"offer {offer['id']} failed: {e}")
    if iid is None:
        raise SystemExit("all offers failed")
    rec = {"instance_id": iid, "run": args.run, "gpu": offer["gpu_name"], "dph": offer["dph_total"],
           "machine_id": offer.get("machine_id"),
           "cap_hours": args.cap_hours, "started_at": time.time(), "cmd": args.cmd}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
