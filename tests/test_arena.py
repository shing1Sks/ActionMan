import pytest

from arena import mining
from arena.agents import RandomAgent, ScriptedAgent
from arena.controls import Scheme
from arena.mining.world import DIAMOND, World
from arena.run import play


def fixed_clock(monkeypatch, step):
    import arena.run as run
    clock = iter(i * step for i in range(10 ** 7))
    monkeypatch.setattr(run.time, "perf_counter", lambda: next(clock))


def test_world_is_deterministic(monkeypatch):
    a, b = World(3, 1), World(3, 1)
    assert a.grid_rows() == b.grid_rows() and a.zombies == b.zombies
    fixed_clock(monkeypatch, 0.003)                       # same latency every call -> identical replays
    ra = play(ScriptedAgent(), mining, 1, 3)
    fixed_clock(monkeypatch, 0.003)                       # fresh clock: float drift along one long clock shifts ticks
    rb = play(ScriptedAgent(), mining, 1, 3)
    assert [c["call"] for c in ra["calls"]] == [c["call"] for c in rb["calls"]]
    assert ra["world"]["blocks"] == rb["world"]["blocks"]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_perfect_driver_wins_every_level(level):
    for seed in range(3):
        r = play(ScriptedAgent(), mining, level, seed)["result"]
        assert r["outcome"] == "win" and r["invalid"] == 0


def test_level0_diamond_is_close_and_has_no_zombies():
    w = World(0, 0)
    d = w.nearest_diamond()
    assert not w.zombies and abs(d[0] - w.px) + abs(d[1] - w.py) <= 10


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_controls_round_trip(level):
    sc = Scheme(mining.SPEC, level, 1, "mining")
    names = {t["name"] for t in sc.tools()}
    acts = [("move", {"direction": d, "steps": 2}) for d in ("left", "right", "up", "down")]
    acts += [("dig", {"direction": "down"}), ("attack", {"direction": "left"}), ("eat", {})]
    for action in acts:
        call = sc.encode(action)
        assert call["name"] in names
        assert sc.translate(call) == (action, "")


def test_renamed_levels_reject_canonical_names():
    sc = Scheme(mining.SPEC, 3, 0, "mining")
    dig, d = sc.tool["dig"], sc.param["direction"]
    assert dig != "dig" and sc.value["dir"]["down"] != "down"
    assert sc.translate({"name": "dig", "args": {"direction": "down"}})[0] is None
    assert sc.translate({"name": dig, "args": {d: "down"}})[0] is None
    assert sc.translate({"name": dig, "args": {d: sc.value["dir"]["down"]}})[0] == ("dig", {"direction": "down"})


def test_optional_steps_default_and_clip():
    sc = Scheme(mining.SPEC, 0, 0, "mining")
    assert sc.translate({"name": "move", "args": {"direction": "up"}})[0] == ("move", {"direction": "up", "steps": 1})
    assert sc.translate({"name": "move", "args": {"direction": "up", "steps": "9"}})[0] == ("move", {"direction": "up", "steps": 3})


def test_invalid_calls_waste_the_turn_and_are_counted():
    class Bad(RandomAgent):
        def act(self, obs, tools, game):
            return [{"name": "teleport", "args": {}}]
    r = play(Bad(), mining, 0, 0)
    assert r["result"]["outcome"] == "timeout"
    assert r["result"]["invalid"] == r["result"]["turns"]


@pytest.mark.parametrize("style", ["order", "state"])
def test_observer_follows_the_plan(style):
    w, obs = World(0, 1), mining.Observer(0, style)
    kind, args = mining.plan(w)
    text = obs.observe(w)
    assert text and (args.get("direction") is None or args["direction"] in text or "you" in text)


def test_digging_the_diamond_wins():
    w = World(0, 0)
    w.grid[w.py + 1][w.px] = DIAMOND
    w.perform(("dig", {"direction": "down"}))
    assert w.outcome is None
    assert w.perform(("dig", {"direction": "down"})) == "DIAMOND!" and w.outcome == "win"


def test_fast_thinking_is_charged_not_rounded_away(monkeypatch):
    fixed_clock(monkeypatch, 0.002)                       # every act() takes 2 ms

    class Steady(RandomAgent):
        def act(self, obs, tools, game):
            return [{"name": "eat", "args": {}}]
    r = play(Steady(), mining, 0, 0)
    charged = sum(c["t_land"] - c["t_obs"] for c in r["calls"])
    real = sum(c["lat_ms"] for c in r["calls"]) / 10      # ms -> 10 ms ticks
    assert abs(charged - real) <= 1                        # rounding would charge 0 ticks for every 2 ms call
