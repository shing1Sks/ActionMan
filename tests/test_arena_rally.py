import json

import pytest

from arena import rally
from arena.agents import RandomAgent, ScriptedAgent
from arena.controls import Scheme
from arena.rally.world import CRASH_TICKS, MAX_CRASHES, V_HAIR, World
from arena.run import play


def fixed_clock(monkeypatch, step):
    import arena.run as run
    clock = iter(i * step for i in range(10 ** 7))
    monkeypatch.setattr(run.time, "perf_counter", lambda: next(clock))


class Late(ScriptedAgent):
    """The perfect call, but every act() takes `lat` seconds on a fake clock."""
    def __init__(self, monkeypatch, lat):
        import arena.run as run
        self.now = 0.0
        monkeypatch.setattr(run.time, "perf_counter", lambda: self.now)
        self.lat = lat

    def act(self, obs, tools, game):
        self.now += self.lat
        return super().act(obs, tools, game)


def test_world_is_deterministic(monkeypatch):
    a, b = World(2, 1), World(2, 1)
    assert a.pts == b.pts and [c.s for c in a.corners] == [c.s for c in b.corners]
    assert a.pts != World(3, 1).pts
    fixed_clock(monkeypatch, 0.003)
    ra = play(ScriptedAgent(), rally, 1, 2)
    fixed_clock(monkeypatch, 0.003)                       # fresh clock per play
    rb = play(ScriptedAgent(), rally, 1, 2)
    assert [c["call"] for c in ra["calls"]] == [c["call"] for c in rb["calls"]]
    assert ra["frames"] == rb["frames"] and ra["world"]["events"] == rb["world"]["events"]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_perfect_driver_wins_every_level(level):
    for seed in range(3):
        rep = play(ScriptedAgent(), rally, level, seed)
        r = rep["result"]
        assert r["outcome"] == "win" and r["invalid"] == 0 and r["crashes"] == 0
        assert r["seconds"] < 20 and len(json.dumps(rep)) < 400_000


def test_level0_is_short_and_gentle():
    w0, w1 = World(0, 0), World(0, 1)
    assert w0.vmax < w1.vmax and len(w0.corners) < len(w1.corners) and w0.length < w1.length
    assert sum(c.sev == 1 for c in w0.corners) == 1 and all(c.sev >= 2 for c in w0.corners if c.sev != 1)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_controls_round_trip(level):
    sc = Scheme(rally.SPEC, level, 1, "rally")
    names = {t["name"] for t in sc.tools()}
    acts = [("steer", {"direction": d, "severity": n}) for d in ("left", "right") for n in (1, 4, 6)]
    acts += [("brake", {}), ("throttle", {})]
    for action in acts:
        call = sc.encode(action)
        assert call["name"] in names
        assert sc.translate(call) == (action, "")


def test_renamed_values_and_severity_clip():
    sc = Scheme(rally.SPEC, 3, 0, "rally")
    steer, d, n = sc.tool["steer"], sc.param["direction"], sc.param["severity"]
    assert steer != "steer" and sc.value["dir"]["left"] != "left"
    assert sc.translate({"name": steer, "args": {d: "left", n: 3}})[0] is None
    assert sc.translate({"name": steer, "args": {d: sc.value["dir"]["left"], n: "9"}})[0] == ("steer", {"direction": "left", "severity": 6})


def test_wrong_or_missing_call_crashes_at_the_corner():
    w = World(0, 1)
    c = w.corners[0]
    w.perform(("steer", {"direction": "left" if c.dir == "right" else "right", "severity": c.sev}))
    while not w.events:
        w.perform(("throttle", {}))
    assert w.events[0].get("crash") == "line" and w.crashes == 1 and w.v == 0
    t0 = w.t
    w.advance(CRASH_TICKS)
    assert w.v > 0 and w.t == t0 + CRASH_TICKS


def test_severity_within_one_passes():
    w = World(0, 1)
    c = w.corners[0]
    w.perform(("steer", {"direction": c.dir, "severity": c.sev + (1 if c.sev < 6 else -1)}))
    while not w.events:
        w.perform(("throttle", {}))
    assert "crash" not in w.events[0]


def test_hairpin_needs_the_brake():
    w = World(0, 0)
    i = next(i for i, c in enumerate(w.corners) if c.sev == 1)
    w.corners, w.ci = w.corners[i:], 0                    # skip straight to the hairpin
    w.perform(("steer", {"direction": w.corners[0].dir, "severity": 1}))
    while not w.events:
        w.perform(("throttle", {}))
    assert w.events[0]["crash"] == "speed"
    w = World(0, 0)
    w.corners, w.ci = w.corners[i:], 0
    w.perform(("steer", {"direction": w.corners[0].dir, "severity": 1}))
    w.perform(("brake", {}))
    while not w.events:
        w.perform(("steer", {"direction": w.corners[0].dir, "severity": 1}))
    assert "crash" not in w.events[0] and w.v <= V_HAIR


def test_too_many_crashes_retire_the_car():
    class Mute(RandomAgent):
        def act(self, obs, tools, game):
            return [{"name": "throttle", "args": {}}]
    r = play(Mute(), rally, 1, 0)["result"]
    assert r["outcome"] == "dead" and r["crashes"] == MAX_CRASHES


def test_latency_decides_the_stage(monkeypatch):
    fast = play(Late(monkeypatch, 0.036), rally, 1, 0)["result"]
    slow = play(Late(monkeypatch, 1.3), rally, 1, 0)["result"]
    assert fast["outcome"] == "win" and fast["crashes"] == 0
    assert slow["crashes"] >= 3


def test_observer_follows_the_plan():
    w, obs = World(1, 1), rally.Observer(1)
    for _ in range(400):
        kind, args = rally.plan(w)
        text = obs.observe(w)
        if kind == "steer":
            assert args["direction"] in text.lower() and str(args["severity"]) in text
        elif kind == "brake":
            assert "brake" in text.lower()
        else:
            assert "flat out" in text.lower() or "full throttle" in text.lower()
        w.perform((kind, args))


def test_fast_thinking_is_charged_not_rounded_away(monkeypatch):
    fixed_clock(monkeypatch, 0.002)
    r = play(ScriptedAgent(), rally, 0, 0)
    charged = sum(c["t_land"] - c["t_obs"] for c in r["calls"])
    real = sum(c["lat_ms"] for c in r["calls"]) / 10
    assert abs(charged - real) <= 1
