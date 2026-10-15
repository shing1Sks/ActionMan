"""Thin wrapper over the vastai CLI (--raw JSON). Never prints API keys."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

VASTAI = shutil.which("vastai") or os.path.expanduser("~/.local/bin/vastai.exe")


def _run(*args: str, raw: bool = True):
    cmd = [VASTAI, *args] + (["--raw"] if raw else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''} failed: {out.stderr[-400:]}")
    if not raw:
        return out.stdout
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return out.stdout


def credit() -> float:
    return float(_run("show", "user").get("credit", 0.0))


def instances() -> list[dict]:
    return _run("show", "instances") or []


def offers(query: str, order: str = "dph", interruptible: bool = False) -> list[dict]:
    args = ["search", "offers", query, "-o", order]
    if interruptible:
        args.append("-i")
    return _run(*args) or []


def create(offer_id: int, image: str, disk: int, env: dict, onstart_path: str, label: str) -> int:
    env_str = " ".join(f"-e {k}={v}" for k, v in env.items())
    cmd = [VASTAI, "create", "instance", str(offer_id), "--image", image, "--disk", str(disk), "--env", env_str,
           "--onstart", onstart_path, "--ssh", "--direct", "--label", label, "--cancel-unavail", "--raw"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        res = json.loads(out.stdout)
    except json.JSONDecodeError:
        res = None
    if isinstance(res, dict) and res.get("success"):
        return int(res["new_contract"])
    secret = env.get("HF_TOKEN") or "<none>"
    raise RuntimeError(f"create failed: {(out.stdout + out.stderr).replace(secret, '<redacted>')[:300]}")


def destroy(instance_id: int):
    return _run("destroy", "instance", str(instance_id), "-y", raw=False)


def logs(instance_id: int, tail: int = 60) -> str:
    return _run("logs", str(instance_id), "--tail", str(tail), raw=False)
