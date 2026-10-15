"""Run arena races and write replays.
python -m arena.run --game mining --levels 0 1 2 3 --seeds 0 1 2 --agents scripted random
Each turn: the game describes what's ahead -> the driver thinks (real wall time; the world keeps moving)
-> its first call lands -> the action costs one tick. The clock is the model."""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import statistics
import time
from types import SimpleNamespace

from .controls import Scheme, call_text

try:
    from actionman import beat                    # touches $HEARTBEAT on GPU boxes so the watchdog sees us alive
except ImportError:
    def beat():
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAYS = os.path.join(HERE, "replays")
GAMES = ["mining", "rally", "cycling"]


def load_game(name: str):
    return importlib.import_module(f".{name}", __package__)


def make_agent(name: str, ckpt: str | None = None):
    from . import agents
    if name == "scripted":
        return agents.ScriptedAgent()
    if name == "random":
        return agents.RandomAgent()
    if name == "actionman":
        from .am_agent import ActionManAgent
        return ActionManAgent(ckpt) if ckpt else ActionManAgent()
    if name == "qwen":
        from .qwen_agent import QwenAgent
        return QwenAgent(ckpt) if ckpt else QwenAgent()
    if name.startswith("openai"):
        from .api_agents import OpenAIAgent
        return OpenAIAgent(*name.split(":")[1:2])
    raise SystemExit(f"unknown agent {name}")


def play(agent, game_mod, level: int, seed: int, **obs_kw) -> dict:
    w, sc = game_mod.World(seed, level), Scheme(game_mod.SPEC, level, seed, game_mod.NAME)
    obs_src, tps = game_mod.Observer(seed, **obs_kw), game_mod.TPS
    game = SimpleNamespace(world=w, scheme=sc, goal=game_mod.GOAL, plan=game_mod.plan)
    tools, log, debt = sc.tools(), [], 0.0          # debt: unspent thinking time in ticks (fast calls add up, never round to 0)
    while not w.outcome:
        beat()
        obs, t_obs = obs_src.observe(w), w.t
        t0, err = time.perf_counter(), ""
        try:
            calls = agent.act(obs, tools, game) or []
        except Exception as e:                     # a crashing agent just wastes the turn
            calls, err = [], f"agent error: {e}"[:200]
        lat = time.perf_counter() - t0
        debt += lat * tps
        ticks = int(debt)
        debt -= ticks
        w.advance(ticks)                           # the world moves while the driver thinks
        call = calls[0] if calls else None
        action, why = sc.translate(call) if not w.outcome else (None, "")
        t_land = w.t
        result = w.perform(action) if not w.outcome else "game over while thinking"
        ended = result == "game over while thinking"         # died or timed out mid-think: not the driver's error
        log.append({"t_obs": t_obs, "t_land": t_land, "t_done": w.t, "lat_ms": round(lat * 1000, 1), "obs": obs,
                    "call": call_text(call), "extra_calls": max(0, len(calls) - 1), "ok": action is not None or ended,
                    "err": "" if ended else (err or why), "result": result})
    lats = [c["lat_ms"] for c in log]
    return {"game": game_mod.NAME, "agent": agent.name, "label": agent.label, "note": agent.note, "level": level, "seed": seed, **obs_kw,
            "tps": tps, "tools": tools, "renames": sc.renames(), "world": w.replay(), "frames": w.frames, "calls": log,
            "result": {"outcome": w.outcome, "seconds": round(w.t / tps, 2), "turns": len(log),
                       "invalid": sum(not c["ok"] for c in log), "lat_ms_p50": round(statistics.median(lats), 1),
                       "lat_ms_mean": round(statistics.mean(lats), 1), "think_s": round(sum(lats) / 1000, 2), **w.summary()}}


def write_viewer_bundle(src: str = REPLAYS):
    reps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(src, "*.json")))]
    with open(os.path.join(HERE, "viewer", "replays.js"), "w") as f:
        f.write("window.ARENA_REPLAYS = " + json.dumps(reps, separators=(",", ":")) + ";\n")
    return len(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["mining"], choices=GAMES)
    ap.add_argument("--levels", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--agents", nargs="+", default=["scripted", "random"])
    ap.add_argument("--ckpt", default=None, help="checkpoint dir for actionman / qwen")
    ap.add_argument("--out", default=REPLAYS)
    ap.add_argument("--style", default=None, help="observation style, for games that have more than one (mining: order | state)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.agents:
        agent = make_agent(name, args.ckpt)
        for g in args.games:
            gm = load_game(g)
            for level in args.levels:
                for seed in args.seeds:
                    rep = play(agent, gm, level, seed, **({"style": args.style} if args.style and g == "mining" else {}))
                    json.dump(rep, open(os.path.join(args.out, f"{g}_L{level}_s{seed}_{agent.name}.json"), "w"))
                    r = rep["result"]
                    print(f"{g:8s} L{level} seed {seed} {agent.label:16s} {r['outcome']:8s} {r['seconds']:7.2f}s "
                          f"turns {r['turns']:4d} invalid {r['invalid']:3d} think {r['think_s']:6.2f}s p50 {r['lat_ms_p50']} ms", flush=True)
    if args.out == REPLAYS:
        print("viewer bundle:", write_viewer_bundle(), "replays")


if __name__ == "__main__":
    main()
