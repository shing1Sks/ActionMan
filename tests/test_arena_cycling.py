import json

import pytest

from arena import cycling
from arena.agents import RandomAgent, ScriptedAgent
from arena.controls import Scheme
from arena.cycling.world import CRASH, SAFE, World
from arena.run import play


def fixed_clock(monkeypatch, step):
    import arena.run as run
    clock = iter(i * step for i in range(10 ** 7))
    monkeypatch.setattr(run.time, "perf_counter", lambda: next(clock))


def ride_to(w, x, v):
    w.x, w.v = x, v


def test_world_is_deterministic(monkeypatch):
    a, b = World(2, 1), World(2, 1)
    assert a.segs == b.segs and a.forks == b.forks and a.hazards == b.hazards
    fixed_clock(monkeypatch, 0.003)
    ra = play(ScriptedAgent(), cycling, 1, 2)
    fixed_clock(monkeypatch, 0.003)
    rb = play(ScriptedAgent(), cycling, 1, 2)
    assert [c["call"] for c in ra["calls"]] == [c["call"] for c in rb["calls"]] and ra["frames"] == rb["frames"]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_perfect_driver_wins_every_level(level):
    for seed in range(3):
        rep = play(ScriptedAgent(), cycling, level, seed)
        r = rep["result"]
        assert r["outcome"] == "win" and r["invalid"] == 0 and r["crashes"] == 0 and r["wrong_turns"] == 0
        assert len(json.dumps(rep)) < 400_000


def test_level0_is_flat_with_one_fork():
    w = World(0, 0)
    assert all(g == 0 for _, _, g in w.segs) and len(w.forks) == 1 and not w.hazards


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_controls_round_trip(level):
    sc = Scheme(cycling.SPEC, level, 1, "cycling")
    names = {t["name"] for t in sc.tools()}
    acts = [("pedal", {}), ("brake", {}), ("shift", {"direction": "up"}), ("shift", {"direction": "down"}),
            ("turn", {"direction": "left"}), ("turn", {"direction": "right"})]
    for action in acts:
        call = sc.encode(action)
        assert call["name"] in names
        assert sc.translate(call) == (action, "")


def test_renamed_values_reject_plain_words():
    sc = Scheme(cycling.SPEC, 3, 0, "cycling")
    turn, d = sc.tool["turn"], sc.param["direction"]
    assert turn != "turn" and sc.value["side"]["left"] != "left"
    assert sc.translate({"name": turn, "args": {d: "left"}})[0] is None
    assert sc.translate({"name": turn, "args": {d: sc.value["side"]["left"]}})[0] == ("turn", {"direction": "left"})


def test_more_strokes_more_speed_with_diminishing_returns():
    def speed_after(every):                   # one stroke every `every` ticks for 3 s on flat road
        w = World(0, 0)
        for t in range(300):
            w.perform(("pedal", {})) if t % every == 0 else w.advance(1)
        return w.v
    fast, mid, slow = speed_after(1), speed_after(4), speed_after(20)
    assert fast > mid > slow and fast - mid < mid - slow


def test_wrong_fork_costs_distance():
    w = World(0, 0)
    f = w.forks[0]
    wrong = "left" if f["c"] == "right" else "right"
    ride_to(w, f["x"] - 5, 10.0)
    assert w.perform(("turn", {"direction": wrong})).startswith("taking")
    w.advance(100)
    assert w.wrong == 1 and w.x == f["x"] and w.detour > 0
    w2 = World(0, 0)
    ride_to(w2, f["x"] - 5, 10.0)
    w2.perform(("turn", {"direction": f["c"]}))
    w2.advance(100)
    assert w2.wrong == 0 and w2.x > f["x"]


def test_unbraked_hazard_crashes_and_braking_does_not():
    w = World(0, 1)
    h = w.hazards[0]
    ride_to(w, h["x"] - 2, 12.0)
    w.advance(30)
    assert w.crashes == 1 and w.v == 0 and w.stun > 0
    assert w.perform(("pedal", {})) == "still on the ground"
    w.advance(CRASH)
    assert w.stun == 0
    w2 = World(0, 1)
    ride_to(w2, h["x"] - 20, 12.0)
    assert w2.perform(("brake", {})).startswith("brakes set")
    w2.advance(500)
    assert w2.crashes == 0 and w2.x > h["x"] + h["l"]


def test_slow_riders_pass_hazards_safely():
    w = World(0, 1)
    h = w.hazards[0]
    ride_to(w, h["x"] - 1, SAFE - 1)
    w.advance(50)
    assert w.crashes == 0


def test_gear_matters_on_climbs():
    w = World(0, 1)
    a, _, g = next(s for s in w.segs if s[2] > 0)
    speeds = {}
    for gear in ("low", "high"):
        w = World(0, 1)
        ride_to(w, a + 1, 4.0)
        w.gear = gear
        for _ in range(100):
            w.perform(("pedal", {}))
        speeds[gear] = w.v
    assert speeds["low"] > speeds["high"] * 1.5


def test_observer_follows_the_plan():
    w, obs = World(0, 1), cycling.Observer(0)
    h = w.hazards[0]
    ride_to(w, h["x"] - 20, 10.0)
    assert cycling.plan(w) == ("brake", {})
    assert str(20) in obs.observe(w)
    w.armed = True
    kind, args = cycling.plan(w)
    text = obs.observe(w)
    if kind == "turn":
        assert args["direction"] in text


def test_invalid_calls_waste_the_turn_and_are_counted():
    class Bad(RandomAgent):
        def act(self, obs, tools, game):
            return [{"name": "teleport", "args": {}}]
    r = play(Bad(), cycling, 0, 0)
    assert r["result"]["outcome"] == "timeout" and r["result"]["invalid"] == r["result"]["turns"]


def test_fast_thinking_is_charged_not_rounded_away(monkeypatch):
    fixed_clock(monkeypatch, 0.002)
    r = play(ScriptedAgent(), cycling, 0, 0)
    charged = sum(c["t_land"] - c["t_obs"] for c in r["calls"])
    real = sum(c["lat_ms"] for c in r["calls"]) / 10
    assert abs(charged - real) <= 1


def test_slower_drivers_finish_later(monkeypatch):
    times = []
    for step in (0.0, 0.036, 0.2):
        fixed_clock(monkeypatch, step)
        times.append(play(ScriptedAgent(), cycling, 1, 0)["result"]["seconds"])
    assert times[0] < times[1] < times[2]
