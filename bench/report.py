"""Benchmark report: small JSON summaries (docs/results/bench/) + figures (docs/results/figures/) from the pulled runs.

python bench/report.py --b1 runs/bench/dl/runs/b1-bfcl4 --b2 runs/bench/dl/runs/b2-latency3 --luna <score dir> \
    --lat-scored runs/bench/lat_scored.json --cpu runs/bench/cpu_laptop*.json
Prints the markdown tables used in docs/results/benchmark.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]
from actionman.evaluate import ece  # noqa: E402

OUT = os.path.join(ROOT, "docs", "results", "bench")
FIG = os.path.join(ROOT, "docs", "results", "figures")
NL = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple"]
LIVE = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
HALL = ["irrelevance", "live_irrelevance", "live_relevance"]
N = {"simple_python": 400, "simple_java": 100, "simple_javascript": 50, "multiple": 200, "parallel": 200,
     "parallel_multiple": 200, "irrelevance": 240, "live_simple": 258, "live_multiple": 1053, "live_parallel": 16,
     "live_parallel_multiple": 24, "live_irrelevance": 884, "live_relevance": 16}


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1)


# ---------- Table 1: BFCL v4 ----------

def score_files(score_dir):
    """{model_dir: {category: (accuracy %, {failing ids})}} from the official per-category score files."""
    out = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(score_dir, "*", "*", "BFCL_v4_*_score.json")):
        model = p.replace("\\", "/").split("/")[-3]
        cat = os.path.basename(p)[len("BFCL_v4_"):-len("_score.json")]
        lines = [json.loads(l) for l in open(p, encoding="utf-8")]
        out[model][cat] = (100 * lines[0]["accuracy"], {r["id"] for r in lines[1:]}, lines[1:])
    return out


def aggregates(cats: dict) -> dict:
    """Leaderboard formulas (eval_runner_helper): NL = mean(simple(py,java,js), multiple, parallel, par-multiple);
    Live = count-weighted; Irrelevance = mean(irrelevance, live_irrelevance). Plus our Python-only NL."""
    a = {c: cats[c] for c in cats}
    g = lambda c: a.get(c)  # noqa: E731
    out = {}
    if all(g(c) is not None for c in NL):
        simple = statistics.mean([g("simple_python"), g("simple_java"), g("simple_javascript")])
        out["Non-Live"] = statistics.mean([simple, g("multiple"), g("parallel"), g("parallel_multiple")])
        out["Non-Live (Python only)"] = statistics.mean([g("simple_python"), g("multiple"), g("parallel"), g("parallel_multiple")])
    if all(g(c) is not None for c in LIVE):
        out["Live"] = sum(g(c) * N[c] for c in LIVE) / sum(N[c] for c in LIVE)
    if g("irrelevance") is not None and g("live_irrelevance") is not None:
        out["Irrelevance"] = (g("irrelevance") + g("live_irrelevance")) / 2
    if g("live_relevance") is not None:
        out["Relevance"] = g("live_relevance")
    return out


def bootstrap_ci(correct: list[int], n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(correct)
    ms = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return 100 * ms[int(0.025 * n_boot)], 100 * ms[int(0.975 * n_boot)]


# ---------- extras (ActionMan) ----------

def call_conf(pred):
    cs = [min([p["conf_tool"]] + list(p["conf_args"].values())) for p in pred]
    return min(cs) if cs else 1.0      # no call: the abstention has no score; evaluate.ece uses 1.0 too


def coverage_curve(pairs):
    """pairs (conf, correct). Answer only the most confident fraction; -> [(coverage, accuracy)]."""
    s = sorted(pairs, key=lambda x: -x[0])
    out, hit = [], 0
    for i, (_, y) in enumerate(s, 1):
        hit += y
        out.append((i / len(s), hit / i))
    return out


def at_cov(curve, cov):
    return 100 * min(curve, key=lambda x: abs(x[0] - cov))[1]


def reliability(pairs, bins=10):
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, y) for c, y in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            rows.append(((lo + hi) / 2, statistics.mean(c for c, _ in sel), statistics.mean(y for _, y in sel), len(sel)))
    return rows


def plot_coverage(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for (label, cur), color in zip(curves.items(), ["#1f4e79", "#c0504d", "#4f7f3a"]):
        ax.plot([100 * c for c, _ in cur], [100 * a for _, a in cur], color=color, lw=1.8, label=label)
    for c in (80, 90):
        ax.axvline(c, color="#999999", lw=0.6, ls=":")
    ax.set_xlabel("coverage (% of prompts answered, most confident first)")
    ax.set_ylabel("accuracy on answered prompts (%)")
    ax.set_xlim(10, 100)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("ActionMan: accuracy vs coverage (abstain below a confidence threshold)", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_reliability(rows, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4), dpi=150)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    ax.plot([r[1] for r in rows], [r[2] for r in rows], "o-", color="#1f4e79", ms=4)
    for r in rows:
        ax.annotate(str(r[3]), (r[1], r[2]), fontsize=6, xytext=(3, -8), textcoords="offset points", color="#555555")
    ax.set_xlabel("mean confidence (min of tool / arg confidence)")
    ax.set_ylabel("full-call accuracy")
    ax.set_title(f"Reliability, {label}", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------- latency ----------

def lat_runs(b2, extra=()):
    """{arm: {rep: summary + records}} from every lat/*.json of the run (+ extra files, keyed arm@file)."""
    out = collections.defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(b2, "lat", "*.json"))) + list(extra):
        d = json.load(open(p))
        if "meta" not in d:
            continue
        for arm, v in d.items():
            if isinstance(v, dict) and "records" in v:
                cpu = os.path.basename(p).startswith("cpu_")     # box-CPU arms (an llama client may see a GPU)
                key = arm + ("@cpu" if cpu else "") + (f"@{os.path.basename(p)[:-5]}" if p in extra else "")
                out[key][d["meta"].get("rep", "A")] = {
                    **v, "arm": arm, "file": os.path.basename(p), "meta": {**d["meta"], "device": "cpu" if cpu else "cuda"}}
    return out


def plot_latency_vs_size(arms, gold_calls, out_tokens, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    colors = ["#1f4e79", "#c0504d", "#4f7f3a", "#8064a2", "#d08a2c", "#4bacc6"]
    for (label, recs), color in zip(arms.items(), colors):
        by = collections.defaultdict(list)
        for r in recs:
            by[min(gold_calls[r["id"]], 3)].append(r["ms"])
        xs = sorted(by)
        axes[0].plot(xs, [statistics.median(by[x]) for x in xs], "o-", color=color, label=label, lw=1.6, ms=4)
        if out_tokens.get(label):
            axes[1].scatter([out_tokens[label][r["id"]] for r in recs], [r["ms"] for r in recs], s=5, alpha=0.5,
                            color=color, label=label)
    axes[0].set_xticks([1, 2, 3]); axes[0].set_xticklabels(["1", "2", "3+"])
    axes[0].set_xlabel("gold calls in the prompt"); axes[0].set_ylabel("median latency (ms, batch 1)")
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.25, lw=0.5, which="both"); axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Latency vs number of calls (500 BFCL v4 prompts, RTX 4090)", fontsize=9)
    axes[1].set_xlabel("LLM output tokens for the prompt"); axes[1].set_ylabel("latency (ms)")
    axes[1].set_yscale("log"); axes[1].grid(alpha=0.25, lw=0.5, which="both"); axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title("Latency vs output length (ActionMan plotted at the SFT's token count)", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", required=True)
    ap.add_argument("--b2", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="official score dirs; a later dir replaces a model wholesale")
    ap.add_argument("--am-flat", required=True, help="bfcl_actionman.py output used for the ActionMan row")
    ap.add_argument("--lat-scored", required=True, help="bfcl_run.py score-calls output over all latency files")
    ap.add_argument("--gold-calls", required=True, help="json {id: n gold calls} for the 500 latency ids")
    ap.add_argument("--cpu", nargs="*", default=[])
    ap.add_argument("--extra-lat", nargs="*", default=[], help="latency jsons from other boxes (accuracy checks)")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    # Table 1
    sc = {}
    for d in args.scores:
        sc.update(score_files(d))
    table1 = {}
    for model, cats in sc.items():
        accs = {c: v[0] for c, v in cats.items()}
        table1[model] = {"categories": accs, **aggregates(accs)}
    save("bfcl_v4.json", table1)
    print("## Table 1\n| model | " + " | ".join(["Non-Live", "Non-Live (Python only)", "Live", "Irrelevance", "Relevance"])
          + " | " + " | ".join(NL + LIVE + HALL) + " |")
    for m, t in table1.items():
        print(f"| {m} | " + " | ".join(f"{t.get(k, float('nan')):.2f}" for k in ["Non-Live", "Non-Live (Python only)", "Live", "Irrelevance", "Relevance"])
              + " | " + " | ".join(f"{t['categories'].get(c, float('nan')):.1f}" for c in NL + LIVE + HALL) + " |")

    # ActionMan per-category bootstrap CI + error classes
    am = sc.get("actionman-s16", {})
    ci = {}
    for c, (acc, fails, _) in am.items():
        n = N[c]
        ci[c] = bootstrap_ci([0] * len(fails) + [1] * (n - len(fails)))
    etypes = {c: collections.Counter(r.get("error_type", "?").split(":")[0] + ":" + r.get("error_type", "?").split(":")[-1]
                                     for r in v[2]) for c, v in am.items()}
    save("actionman_bfcl_ci_errors.json", {"ci95": ci, "official_error_types": etypes})
    print("\n## ActionMan BFCL v4 bootstrap 95% CI\n" + "\n".join(f"- {c}: {am[c][0]:.1f} [{lo:.1f}, {hi:.1f}]" for c, (lo, hi) in ci.items()))

    # errclass on wrong args (AST categories)
    from actionman.errclass import classify_arg
    from actionman.scoring import value_match
    ecl = collections.defaultdict(collections.Counter)
    for c, (_, _, rows) in am.items():
        if c in HALL:
            continue
        for r in rows:
            dec, gold = r.get("model_result_decoded"), r.get("possible_answer")
            if not isinstance(dec, list) or not isinstance(gold, list):
                ecl[c]["undecodable/empty"] += 1
                continue
            pn = sorted(next(iter(d)) for d in dec if isinstance(d, dict) and d)
            gn = sorted(next(iter(g)) for g in gold)
            if pn != gn:
                ecl[c]["wrong_tool_or_count"] += 1
                continue
            pool = collections.defaultdict(list)
            for d in dec:
                (n, a), = d.items(); pool[n].append(a)
            q = r["prompt"]["question"][0][-1]["content"] if isinstance(r.get("prompt"), dict) else ""
            for g in gold:
                (n, acc), = g.items()
                got = pool[n].pop(0) if pool[n] else {}
                for k, vals in acc.items():
                    pv = got.get(k)
                    if pv is None and "" in vals:
                        continue
                    if pv is not None and any(value_match(pv, v) for v in vals if v != ""):
                        continue
                    ecl[c][classify_arg(pv, vals, q)] += 1
                for k in got:
                    if k not in acc:
                        ecl[c]["extra_arg"] += 1
    save("actionman_bfcl_errclass.json", ecl)
    print("\n## errclass\n" + "\n".join(f"- {c}: {dict(v.most_common())}" for c, v in ecl.items()))

    # status (fit / truncated / too long)
    st = collections.Counter(json.loads(l)["status"] for l in open(args.am_flat, encoding="utf-8"))
    print(f"\n## ActionMan prompt status: {dict(st)}")

    # own eval: leniency gap + calibration + coverage
    strict = json.load(open(os.path.join(args.b1, "am_own_strict.json")))
    own_preds = [json.loads(l) for l in open(os.path.join(args.b1, "am_own_preds.jsonl"), encoding="utf-8")]
    # calibration: prompts where ActionMan called (its confidence is defined there);
    # coverage: prompts that need a call, a no-call answer ranks lowest (it already is an abstention)
    called_own = [(call_conf(r["pred"]), int(r["full"])) for r in own_preds if r["pred"]]
    need_own = [(call_conf(r["pred"]) if r["pred"] else 0.0, int(r["full"])) for r in own_preds if r["gold"]]
    am_flat = {json.loads(l)["id"]: json.loads(l) for l in open(args.am_flat, encoding="utf-8")}
    pairs_bfcl, need_bfcl = [], []
    for c, (_, fails, _) in am.items():
        if c in ("simple_java", "simple_javascript"):
            continue
        for i, r in am_flat.items():
            if i.rsplit("_", 1)[0] == c:
                if r["conf"] is not None:
                    pairs_bfcl.append((r["conf"], int(i not in fails)))
                if c not in HALL:
                    need_bfcl.append((r["conf"] if r["conf"] is not None else 0.0, int(i not in fails)))
    pairs_own = called_own
    cur_own, cur_bfcl = coverage_curve(need_own), coverage_curve(need_bfcl)
    extras = {"leniency": strict,
              "ece": {"own_eval_lenient_called": ece(pairs_own), "bfcl_v4_python_called": ece(pairs_bfcl),
                      "n": [len(pairs_own), len(pairs_bfcl)]},
              "coverage": {k: {f"{int(c * 100)}%": at_cov(cur, c) for c in (1.0, 0.9, 0.8)}
                           for k, cur in (("own_eval", cur_own), ("bfcl_v4_python", cur_bfcl))},
              "reliability_own": reliability(pairs_own), "reliability_bfcl": reliability(pairs_bfcl),
              "seed_band_own_eval": {"v3_seeds": [78.6, 78.8, 77.5], "v31_seeds": [77.6, 79.1, 78.9], "release_5ep": 79.7}}
    save("extras.json", extras)
    plot_coverage({f"own eval, prompts needing a call (n={len(need_own)}, lenient scorer)": cur_own,
                   f"BFCL v4 python AST, non-live + live (n={len(need_bfcl)}, official)": cur_bfcl},
                  os.path.join(FIG, "accuracy_vs_coverage.png"))
    plot_reliability(extras["reliability_bfcl"], os.path.join(FIG, "reliability_bfcl_v4.png"), "BFCL v4 (official checker)")
    print("\n## extras", json.dumps({k: v for k, v in extras.items() if not k.startswith("reliability")}, indent=1))

    own = {os.path.basename(p)[4:-5]: json.load(open(p)) for p in glob.glob(os.path.join(args.b1, "own_*.json"))}
    save("own_eval_official.json", {"actionman-s16": strict, **own})
    print("\n## own eval (official checker): " + json.dumps({m: {k: round(v.get("strict_pct", 0), 1) for k, v in d.items()} for m, d in own.items()}))

    # Table 2: GPU latency
    lat, scored = lat_runs(args.b2, args.extra_lat), json.load(open(args.lat_scored))
    gold_calls = json.load(open(args.gold_calls))
    t2 = {}
    for arm, reps in lat.items():
        row = {}
        for rep, v in reps.items():
            acc = scored.get(f"{v['file']}:{v['arm']}", {}).get("acc")
            row[rep] = {k: v[k] for k in ("n", "p50_ms", "p95_ms", "mean_ms")} | {
                "acc": acc, "out_tokens": statistics.mean(r["out_tokens"] for r in v["records"]),
                "device": v["meta"].get("device"), "threads": v["meta"].get("threads")}
        t2[arm] = row
    meta = next(iter(next(iter(lat.values())).values()))["meta"] if lat else {}
    save("latency.json", {"meta": meta, "arms": t2})
    print("\n## Table 2/3 latency")
    for arm, row in t2.items():
        for rep, r in sorted(row.items()):
            print(f"| {arm} | {rep} | {r['device']} | {r['n']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['mean_ms']:.1f} | "
                  f"{(r['acc'] or float('nan')):.1f} | {r['out_tokens']:.0f} |")
    # buckets and figure from rep A (gpu arms with 500 prompts)
    figarms, toks = {}, {}
    for arm, reps in lat.items():
        v = reps.get("A")
        if v and v["meta"].get("device") == "cuda" and v["n"] == 500 and arm in (
                "actionman_compiled_T3", "actionman_eager_T3", "vllm_qwen_sft", "vllm_qwen_sft_seqs1",
                "vllm_qwen3_0.6b_public", "llamacpp_q8_gpu"):
            figarms[arm] = v["records"]
    if "vllm_qwen_sft" in figarms:
        sft_tok = {r["id"]: r["out_tokens"] for r in figarms["vllm_qwen_sft"]}
        for arm, recs in figarms.items():
            toks[arm] = sft_tok if arm.startswith("actionman") else {r["id"]: r["out_tokens"] for r in recs}
    if figarms:
        plot_latency_vs_size(figarms, gold_calls, toks, os.path.join(FIG, "latency_vs_output_size.png"))
        bk = {arm: {str(b): statistics.median([r["ms"] for r in recs if min(gold_calls[r["id"]], 3) == b]) for b in (1, 2, 3)}
              for arm, recs in figarms.items()}
        save("latency_buckets.json", {"by_gold_calls_median_ms": bk,
                                      "bucket_n": collections.Counter(min(v, 3) for v in gold_calls.values())})
        print("\n## buckets (median ms by gold calls 1/2/3+)\n" + json.dumps(bk, indent=1))

    for p in args.cpu:
        d = json.load(open(p))
        print("\n## CPU", p, json.dumps(d.get("meta")))
        for arm, v in d.items():
            if isinstance(v, dict) and "p50_ms" in v:
                print(f"| {arm} | {v['n']} | {v['p50_ms']:.0f} | {v['p95_ms']:.0f} | {v['mean_ms']:.0f} |")
            if arm == "am_profile":
                print(json.dumps({k: v[k] for k in ("n", "mean_ms_profiled", "stages_ms", "threads")}, indent=1))
                print(v["top_ops"])


if __name__ == "__main__":
    main()
