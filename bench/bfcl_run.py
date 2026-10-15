"""Official BFCL v4 harness (bfcl-eval==2026.3.23) plus our registrations; runs in the bfcl-eval/vLLM venv.

Registered in-process (the installed package is not edited):
  actionman-s16, actionman-s16-last   ActionMan predictions made by bench/bfcl_actionman.py (flat / last-user prompt)
  am-qwen-sft-FC                      our Qwen3-0.6B SFT with its training prompt (HF template, to_openai_tool, no think)
  gpt-6-luna-FC                       GPT-6 Luna via the Responses API, reasoning effort low (laptop only)

python bench/bfcl_run.py dump-entries --out data/bench/bfcl_v4_entries.jsonl
python bench/bfcl_run.py import --preds preds.jsonl --model actionman-s16 --result-dir result
python bench/bfcl_run.py cli generate|evaluate ...          (= `bfcl ...` with the registrations)
python bench/bfcl_run.py gate --score-dir score              (Qwen3-0.6B FC vs leaderboard; exit 1 on fail)
python bench/bfcl_run.py strict --preds <evaluate --dump>    (own 2,340-prompt eval, official checker)
python bench/bfcl_run.py own --model <bfcl model> --out own.jsonl [--local-model-path P]
python bench/bfcl_run.py luna --cap 1.4 --result-dir result_luna
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import csv
import json
import os
import sys
import threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT, os.path.dirname(__file__)]
from bfcl_eval.constants.category_mapping import SINGLE_TURN_CATEGORY  # noqa: E402
from bfcl_eval.constants.enums import Language, ReturnFormat  # noqa: E402
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402
from bfcl_eval.model_handler.api_inference.openai_response import OpenAIResponsesHandler  # noqa: E402
from bfcl_eval.model_handler.base_handler import BaseHandler  # noqa: E402
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler  # noqa: E402
from bfcl_eval.utils import (is_empty_output, is_function_calling_format_output, load_dataset_entry,  # noqa: E402
                             load_ground_truth_entry)
from overrides import override  # noqa: E402

from bfcl_actionman import category, example_to_bfcl  # noqa: E402
from actionman.schema import example_from_dict, from_bfcl, to_openai_tool  # noqa: E402

# public leaderboard (gorilla.cs.berkeley.edu data_overall.csv, fetched 2026-10-15): Non-Live, Live, Irrelevance
LEADERBOARD = {"Qwen/Qwen3-0.6B-FC": (71.79, 56.62, 80.84), "MadeAgents/Hammer2.1-0.5b": (65.98, 54.63, 80.79),
               "ibm-granite/granite-4.0-350m": (67.92, 46.11, 60.84), "Qwen/Qwen3-1.7B-FC": (82.92, 74.61, 76.54)}
TOL = {"Non-Live": 1.0, "Irrelevance": 2.0, "Live": 2.0}   # Live is a secondary check (not gating)
PRICE = (0.10, 0.50)   # GPT-6 Luna $/M input, output


class ActionManHandler(BaseHandler):
    """Results are written by `import`; this only decodes [{name: json args}] (an empty list = no call)."""
    def decode_ast(self, result, language, has_tool_call_tag):
        return [{n: json.loads(a)} for c in result for n, a in c.items()]

    def decode_execute(self, result, has_tool_call_tag):
        raise NotImplementedError


class QwenSFTHandler(QwenFCHandler):
    """Our SFT was trained on the HF chat template with to_openai_tool tools and enable_thinking=False."""
    @override
    def _format_prompt(self, messages, function):
        tools = [to_openai_tool(t) for t in from_bfcl({"question": [[]], "function": function, "id": ""}, None, "").tools]
        return self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                  enable_thinking=False, tokenize=False)


class LunaHandler(OpenAIResponsesHandler):
    def _query_FC(self, inference_data):
        kw = {"input": inference_data["message"], "model": self.model_name, "store": False, "reasoning": {"effort": "low"}}
        if inference_data["tools"]:
            kw["tools"] = inference_data["tools"]
        return self.generate_with_backoff(**kw)


def register():
    def cfg(name, display, handler, lic, dot=False):
        return ModelConfig(model_name=name, display_name=display, url="", org="ActionMan bench", license=lic,
                           model_handler=handler, input_price=None, output_price=None, is_fc_model=True,
                           underscore_to_dot=dot)
    MODEL_CONFIG_MAPPING["actionman-s16"] = cfg("actionman-s16", "ActionMan s16-v3-5epb", ActionManHandler, "unreleased")
    MODEL_CONFIG_MAPPING["actionman-s16-last"] = cfg("actionman-s16-last", "ActionMan s16 (last-user)", ActionManHandler, "unreleased")
    MODEL_CONFIG_MAPPING["actionman-s16-compiled"] = cfg("actionman-s16-compiled", "ActionMan s16 (compiled)", ActionManHandler, "unreleased")
    MODEL_CONFIG_MAPPING["am-qwen-sft-FC"] = cfg("Qwen/Qwen3-0.6B", "Qwen3-0.6B SFT (ours)", QwenSFTHandler, "unreleased")
    MODEL_CONFIG_MAPPING["gpt-6-luna-FC"] = cfg("gpt-6-luna", "GPT-6 Luna (FC, effort low)", LunaHandler, "Proprietary", True)


def handler_for(model):
    c = MODEL_CONFIG_MAPPING[model]
    return c.model_handler(model_name=c.model_name, temperature=0.001, registry_name=model, is_fc_model=c.is_fc_model)


def result_dir(p):
    from bfcl_eval.constants.eval_config import PROJECT_ROOT
    return PROJECT_ROOT / p


def dump_entries(args):
    with open(args.out, "w", encoding="utf-8") as f:
        for c in SINGLE_TURN_CATEGORY:
            for e in load_dataset_entry(c):
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("wrote", args.out)


def import_preds(args):
    h = handler_for(args.model)
    rows = [json.loads(l) for l in open(args.preds, encoding="utf-8")]
    h.write([{"id": r["id"], "result": r["result"]} for r in rows], result_dir=result_dir(args.result_dir))
    st = {}
    for r in rows:
        st.setdefault(category(r["id"]), {}).setdefault(r["status"], 0)
        st[category(r["id"])][r["status"]] += 1
    print(json.dumps(st))


def read_scores(score_dir, display):
    out = {}
    for fn in ("data_non_live.csv", "data_live.csv", "data_overall.csv"):
        p = os.path.join(score_dir, fn)
        if os.path.exists(p):
            for row in csv.DictReader(open(p, encoding="utf-8")):
                if row["Model"] == display:
                    out[fn] = row
    return out


def pct(s):
    try:
        return float(str(s).strip("%"))
    except ValueError:
        return None


def gate(args):
    """Every --models entry within TOL of the leaderboard on Non-Live and Irrelevance (Live reported only)."""
    oks = []
    for m in args.models.split(","):
        s = read_scores(args.score_dir, MODEL_CONFIG_MAPPING[m].display_name)
        nl, lv = s.get("data_non_live.csv", {}), s.get("data_live.csv", {})
        irr = [pct(nl.get("Irrelevance Detection")), pct(lv.get("Irrelevance Detection"))]
        got = {"Non-Live": pct(nl.get("Non-Live Overall Acc")), "Live": pct(lv.get("Live Overall Acc")),
               "Irrelevance": sum(irr) / 2 if None not in irr else None}
        ref = dict(zip(("Non-Live", "Live", "Irrelevance"), LEADERBOARD[m]))
        if all(v is None for v in got.values()):
            print(json.dumps({"model": m, "missing": True}))
            continue
        ok = all(got[k] is not None and abs(got[k] - ref[k]) <= TOL[k] for k in ("Non-Live", "Irrelevance"))
        oks.append(ok)
        print(json.dumps({"model": m, "got": got, "leaderboard": ref, "pass": ok}))
    sys.exit(0 if oks and all(oks) else 1)


def strict_valid(decoded, entry, gold, cat, model):
    """Official single-turn check: irrelevance = no call; otherwise the AST checker."""
    if not gold:
        return is_empty_output(decoded)
    if not is_function_calling_format_output(decoded):
        return False
    return ast_checker(entry["function"], decoded, gold, Language.PYTHON, cat, model)["valid"]


def own_items(data):
    """Our 2,340 eval prompts as (bfcl entry, gold, checker category); BFCL rows reuse the v4 function docs."""
    v4 = {e["id"]: e["function"] for c in ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance")
          for e in load_dataset_entry(c, include_language_specific_hint=False)}
    return [example_to_bfcl(example_from_dict(json.loads(l)), v4) for p in data for l in open(p, encoding="utf-8")]


def strict(args):
    """Rescore an `evaluate.py --dump` file (lenient `full` per prompt) with the official checker."""
    from bfcl_actionman import to_result
    from actionman.schema import Call
    items = {e["orig_id"]: (e, g, c) for e, g, c in own_items(args.data)}
    stats, rows = {}, []
    for line in open(args.preds, encoding="utf-8"):
        r = json.loads(line)
        e, g, c = items[r["id"]]
        res = to_result([Call(p["name"], p["args"]) for p in r["pred"]], e["function"], c)
        ok = strict_valid([{n: json.loads(a)} for x in res for n, a in x.items()], e, g, c, "actionman-s16")
        for k in (r["source"], "all"):
            s = stats.setdefault(k, {"n": 0, "lenient": 0, "strict": 0})
            s["n"] += 1; s["lenient"] += bool(r["full"]); s["strict"] += ok
        if r["full"] and not ok:
            rows.append({"id": r["id"], "source": r["source"], "query": e["question"][0][-1]["content"], "pred": r["pred"], "gold": g})
    out = {k: {**s, "lenient_pct": 100 * s["lenient"] / s["n"], "strict_pct": 100 * s["strict"] / s["n"],
               "gap_pts": 100 * (s["lenient"] - s["strict"]) / s["n"]} for k, s in stats.items()}
    json.dump(out, open(args.out, "w"), indent=1)
    with open(args.out.replace(".json", "_strict_only.jsonl"), "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({k: (round(v["lenient_pct"], 1), round(v["strict_pct"], 1)) for k, v in out.items()}))


def own(args):
    """A bfcl-registered model on our 2,340 prompts via its own handler + vLLM; scored with the official checker."""
    from bfcl_eval.constants.eval_config import LOCAL_SERVER_MAX_CONCURRENT_REQUEST
    h, items = handler_for(args.model), own_items(args.data)
    h.spin_up_local_server(num_gpus=1, gpu_memory_utilization=0.9, backend="vllm", skip_server_setup=False,
                           local_model_path=args.local_model_path)
    stats, lock = {}, threading.Lock()

    def run(item):
        e, g, c = item
        try:
            res, _ = h.inference(copy.deepcopy(e), False, False)
            dec = h.decode_ast(res, ReturnFormat.PYTHON, False)
        except Exception:
            dec = []          # undecodable = no call (what the official relevance runner does)
        ok = strict_valid(dec, e, g, c, args.model)
        with lock:
            for k in (e["source"], "all"):
                s = stats.setdefault(k, {"n": 0, "strict": 0})
                s["n"] += 1; s["strict"] += ok
    try:
        with cf.ThreadPoolExecutor(LOCAL_SERVER_MAX_CONCURRENT_REQUEST) as pool:
            list(pool.map(run, items))
    finally:
        h.shutdown_local_server()
    out = {k: {**s, "strict_pct": 100 * s["strict"] / s["n"]} for k, s in stats.items()}
    json.dump(out, open(args.out, "w"), indent=1)
    print(args.model, json.dumps({k: round(v["strict_pct"], 1) for k, v in out.items()}))


def score_calls(args):
    """Official-checker accuracy of latency records ({id, calls: [{name: args}]}) per arm."""
    cache, out = {}, {}

    def gold(eid):
        c = category(eid)
        if c not in cache:
            es = load_dataset_entry(c, include_prereq=False, include_language_specific_hint=False)
            cache[c] = {e["id"]: (e, g["ground_truth"]) for e, g in zip(es, load_ground_truth_entry(c))}
        return cache[c][eid]
    for p in args.lat:
        for arm, v in json.load(open(p)).items():
            if isinstance(v, dict) and "records" in v:
                ok = {r["id"]: strict_valid(r["calls"], *gold(r["id"]), category(r["id"]), "actionman-s16")
                      for r in v["records"]}
                out[f"{os.path.basename(p)}:{arm}"] = {"n": len(ok), "acc": 100 * sum(ok.values()) / len(ok), "ok": ok}
    json.dump(out, open(args.out, "w"))
    print(json.dumps({k: round(v["acc"], 1) for k, v in out.items()}))


def luna(args):
    """GPT-6 Luna on the single-turn categories, cheapest-first, with a hard dollar cap. Key read, never printed."""
    from arena.api_agents import env_key
    os.environ["OPENAI_API_KEY"] = env_key("OPENAI_API_KEY")
    h, out = handler_for("gpt-6-luna-FC"), result_dir(args.result_dir)
    stride = dict(x.split(":") for x in args.stride.split(",")) if args.stride else {}
    done = set()
    for p in out.rglob("*_result.json"):
        done |= {json.loads(l)["id"] for l in open(p)}
    todo = [e for c in args.cats.split(",") for i, e in enumerate(load_dataset_entry(c))
            if i % int(stride.get(c, 1)) == 0 and e["id"] not in done]
    spent, lock, stop = args.spent, threading.Lock(), threading.Event()

    def run(e):
        nonlocal spent
        if stop.is_set():
            return
        try:
            res, meta = h.inference(copy.deepcopy(e), False, False)
        except Exception as ex:   # recorded like `bfcl generate` does
            res, meta = f"Error during inference: {ex}"[:300], {}
        with lock:
            spent += (meta.get("input_token_count", 0) * PRICE[0] + meta.get("output_token_count", 0) * PRICE[1]) / 1e6
            h.write({"id": e["id"], "result": res, **meta}, result_dir=out)
            if spent >= args.cap:
                stop.set()
    with cf.ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(run, todo))
    print(json.dumps({"spent_usd": round(spent, 4), "requested": len(todo), "stopped_at_cap": stop.is_set()}))


def main():
    register()
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from bfcl_eval.__main__ import cli
        sys.argv = ["bfcl"] + sys.argv[2:]
        return cli()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump-entries"); p.add_argument("--out", required=True)
    p = sub.add_parser("import"); p.add_argument("--preds", required=True); p.add_argument("--model", required=True)
    p.add_argument("--result-dir", default="result")
    p = sub.add_parser("gate"); p.add_argument("--score-dir", required=True)
    p.add_argument("--models", default="Qwen/Qwen3-0.6B-FC")
    for name in ("strict", "own"):
        p = sub.add_parser(name); p.add_argument("--data", nargs="+", default=["data/prepared/test.jsonl", "data/prepared/bfcl.jsonl"])
        p.add_argument("--out", required=True)
        if name == "strict":
            p.add_argument("--preds", required=True)
        else:
            p.add_argument("--model", required=True); p.add_argument("--local-model-path", default=None)
    p = sub.add_parser("score-calls"); p.add_argument("--lat", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("luna"); p.add_argument("--cap", type=float, default=1.4); p.add_argument("--spent", type=float, default=0.0)
    p.add_argument("--cats", default=",".join(SINGLE_TURN_CATEGORY)); p.add_argument("--stride", default="")
    p.add_argument("--threads", type=int, default=8); p.add_argument("--result-dir", default="result")
    args = ap.parse_args()
    {"dump-entries": dump_entries, "import": import_preds, "gate": gate, "strict": strict, "own": own, "luna": luna,
     "score-calls": score_calls}[args.cmd](args)


if __name__ == "__main__":
    main()
