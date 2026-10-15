"""Batch-1 latency: ActionMan (T=1,2,3) vs Qwen baseline on the same prompts and GPU.

python bench/latency.py --actionman runs/v1 --qwen runs/qwen --n 300 --out runs/latency.json
Benchmark plan (frozen BFCL v4 ids, end to end, one engine per process; see bench/b2_latency.sh):
python bench/latency.py --ids bench/latency_ids.json --arms am_eager,am_compiled --actionman <ckpt> --out a.json
python bench/latency.py --ids bench/latency_ids.json --arms vllm --llm <hf dir> --label vllm_qwen_sft --out b.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from actionman import beat  # noqa: E402
from actionman.data import load_jsonl  # noqa: E402


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def stats(ms):
    s = sorted(ms)
    return {"n": len(s), "mean_ms": statistics.mean(s), "p50_ms": s[len(s) // 2], "p95_ms": s[int(0.95 * (len(s) - 1))]}


COMPILE = False


def bench_actionman(ckpt, exs, device, warm=10):
    from actionman.evaluate import is_cached, load_model, uses_markers, uses_v3
    from actionman.infer import predict
    model, tok = load_model(ckpt, device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    res, cached, mk, v3 = {}, is_cached(ckpt), uses_markers(ckpt), uses_v3(ckpt)
    for T in (1, 2, 3):
        for ex in exs[:warm]:
            predict(model, tok, ex, T=T, device=device, cached=cached, markers=mk, v3=v3)
        ms, passes = [], []
        for ex in exs:
            beat(); sync(device); t = time.perf_counter()
            p = predict(model, tok, ex, T=T, device=device, cached=cached, markers=mk, v3=v3)
            sync(device); ms.append(1000 * (time.perf_counter() - t)); passes.append(p.passes)
        res[f"actionman{'_cached' if cached else ''}_T{T}"] = {**stats(ms), "avg_forward_passes": statistics.mean(passes)}
    if COMPILE and cached:
        # S6: same predict(), cached encoder passes swapped for padded static-shape cores under torch.compile
        # (mode="reduce-overhead" -> CUDA graphs). One untimed sweep compiles every length bucket first.
        import actionman.cached as C
        eager = (C.encode_context, C.encode_slots)
        static = C.make_static(model.mlm.model)
        sig = lambda p: [(c.name, c.args) for c in p.calls]  # noqa: E731
        try:
            for T in (1, 2, 3):
                C.encode_context, C.encode_slots = eager
                ref = [predict(model, tok, ex, T=T, device=device, cached=cached, markers=mk, v3=v3) for ex in exs]
                C.encode_context, C.encode_slots = static
                for ex in exs:                                   # compile sweep (untimed)
                    beat(); predict(model, tok, ex, T=T, device=device, cached=cached, markers=mk, v3=v3)
                ms, agree = [], 0
                for ex, r in zip(exs, ref):
                    beat(); sync(device); t = time.perf_counter()
                    p = predict(model, tok, ex, T=T, device=device, cached=cached, markers=mk, v3=v3)
                    sync(device); ms.append(1000 * (time.perf_counter() - t))
                    agree += sig(p) == sig(r)
                res[f"actionman_compiled_T{T}"] = {**stats(ms), "same_as_eager_pct": 100 * agree / len(exs)}
                print(T, res[f"actionman_compiled_T{T}"], flush=True)
        finally:
            C.encode_context, C.encode_slots = eager
    # CPU-side featurisation (tokenising tool signatures + query) included in the numbers above;
    # a deployment would cache the catalog tokens, so report it separately.
    from actionman.features import build_pass_a
    from actionman.schema import Example
    fm = []
    for ex in exs:
        t = time.perf_counter()
        build_pass_a(tok, Example(ex.query, ex.tools, [], ex.id, ex.source))
        fm.append(1000 * (time.perf_counter() - t))
    res["actionman_featurize_cpu"] = stats(fm)
    return res


def bench_qwen(ckpt, exs, device, warm=5):
    from baseline.qwen import load, prompt_text
    model, tok = load(ckpt, False, device)
    model.eval()
    ms, new = [], []
    with torch.no_grad():
        for i, ex in enumerate(exs[:warm] + exs):
            enc = tok(prompt_text(tok, ex), return_tensors="pt", add_special_tokens=False).to(device)
            beat(); sync(device); t = time.perf_counter()
            g = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
            sync(device)
            if i >= warm:
                ms.append(1000 * (time.perf_counter() - t)); new.append(g.shape[1] - enc.input_ids.shape[1])
    return {"qwen": {**stats(ms), "avg_new_tokens": statistics.mean(new)}}


# ---------- benchmark-plan arms: frozen BFCL v4 prompt ids, end to end (entry -> parsed calls) ----------
# Every arm times the same span: BFCL entry (messages + function docs) -> tool conversion / templating -> engine ->
# parsed list of calls. Model load, the first WARM prompts and compile / CUDA-graph capture are excluded.

WARM = 20


def run_arm(fn, entries, name, warm_all=False):
    """fn(entry) -> (calls [{name: args}], out_tokens). Untimed warm-up first, then every entry timed."""
    for e in (entries if warm_all else entries[:WARM]):
        beat(); fn(e)
    recs = []
    for e in entries:
        beat(); t = time.perf_counter()
        calls, ntok = fn(e)
        recs.append({"id": e["id"], "ms": 1000 * (time.perf_counter() - t), "calls": calls, "out_tokens": ntok})
    print(name, stats([r["ms"] for r in recs]), flush=True)
    return {**stats([r["ms"] for r in recs]), "records": recs}


def am_arms(ckpt, entries, device, T_list, compiled, mode):
    from bench.bfcl_actionman import fit, entry_to_example, to_result
    from actionman.evaluate import is_cached, load_model, uses_markers, uses_v3
    from actionman.infer import predict
    model, tok = load_model(ckpt, device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    flags = dict(cached=is_cached(ckpt), markers=uses_markers(ckpt), v3=uses_v3(ckpt), device=device)
    if compiled:
        import actionman.cached as C
        C.encode_context, C.encode_slots = C.make_static(model.mlm.model)
    out = {}
    for T in T_list:
        def fn(e, T=T):
            ex, status = fit(tok, entry_to_example(e, mode))
            if status == "too_long":
                return [], 0
            p = predict(model, tok, ex, T=T, **flags)
            sync(device)
            return [{n: json.loads(a)} for c in to_result([c.as_call() for c in p.calls], e["function"], ex.source)
                    for n, a in c.items()], 0
        out[f"actionman_{'compiled' if compiled else 'eager'}_T{T}"] = run_arm(fn, entries, f"am T{T}", warm_all=compiled)
    return out


def am_profile(ckpt, entries, mode, T=3):
    """Where an eager ActionMan call goes (CPU): stage wall times + torch.profiler op table over the given prompts."""
    import collections
    import actionman.cached as C
    import actionman.infer_v3 as I
    from torch.profiler import ProfilerActivity, profile
    from bench.bfcl_actionman import entry_to_example, fit
    from actionman.evaluate import is_cached, load_model, uses_markers, uses_v3
    from actionman.infer import predict
    model, tok = load_model(ckpt, "cpu")
    flags = dict(cached=is_cached(ckpt), markers=uses_markers(ckpt), v3=uses_v3(ckpt), device="cpu")
    st, lens = collections.defaultdict(float), []

    def timed(name, f):
        def g(*a, **k):
            t = time.perf_counter(); r = f(*a, **k); st[name] += 1000 * (time.perf_counter() - t)
            if name == "pass_A_context_encoder":
                lens.append(a[1].shape[1])
            return r
        return g
    orig = (C.encode_context, C.encode_slots, I.build_pass_a_v3, I.build_pass_b_v3)
    C.encode_context, C.encode_slots = timed("pass_A_context_encoder", orig[0]), timed("slot_passes", orig[1])
    I.build_pass_a_v3, I.build_pass_b_v3 = timed("featurize_A", orig[2]), timed("featurize_B", orig[3])
    exs = [fit(tok, entry_to_example(e, mode))[0] for e in entries]
    predict(model, tok, exs[0], T=T, **flags)                       # warm-up
    st.clear(); lens.clear()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for ex in exs:
            beat(); predict(model, tok, ex, T=T, **flags)
    total = 1000 * (time.perf_counter() - t0)
    C.encode_context, C.encode_slots, I.build_pass_a_v3, I.build_pass_b_v3 = orig
    n = len(exs)
    stages = {k: v / n for k, v in st.items()}
    stages["other (heads, decode, python)"] = total / n - sum(stages.values())
    return {"am_profile": {"n": n, "mean_ms_profiled": total / n, "stages_ms": stages, "context_tokens": lens,
                           "threads": torch.get_num_threads(),
                           "top_ops": prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)}}


def llm_prompt_fn(tok_path):
    """HF chat template with the SFT's training tool format, thinking off; same prompt text for every LLM engine."""
    from transformers import AutoTokenizer
    from actionman.schema import from_bfcl, to_openai_tool
    tok = AutoTokenizer.from_pretrained(tok_path)

    def prompt(e):
        tools = [to_openai_tool(t) for t in from_bfcl(e, None, "").tools]
        return tok.apply_chat_template(e["question"][0], tools=tools, add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
    return prompt, tok


def parsed(text):
    from baseline.qwen import parse_calls
    return [{c.name: c.args} for c in parse_calls(text)]


def vllm_arm(model_path, entries, label, max_num_seqs, max_tokens=256, dtype="bfloat16"):
    from vllm import LLM, SamplingParams
    prompt, _ = llm_prompt_fn(model_path)
    llm = LLM(model_path, dtype=dtype, enable_prefix_caching=False, gpu_memory_utilization=0.6,
              max_model_len=8192, max_num_seqs=max_num_seqs, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def fn(e):
        o = llm.generate([prompt(e)], sp, use_tqdm=False)[0].outputs[0]
        return parsed(o.text), len(o.token_ids)
    return {label: run_arm(fn, entries, label)}


def hf_arm(model_path, entries, label, device, max_tokens=256):
    from transformers import AutoModelForCausalLM
    prompt, tok = llm_prompt_fn(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                                 attn_implementation="sdpa").to(device).eval()

    @torch.no_grad()
    def fn(e):
        enc = tok(prompt(e), return_tensors="pt", add_special_tokens=False).to(device)
        g = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        new = g[0, enc.input_ids.shape[1]:]
        return parsed(tok.decode(new, skip_special_tokens=False)), len(new)
    return {label: run_arm(fn, entries, label)}


def llamacpp_arm(tok_path, url, entries, label, max_tokens=256):
    """llama-server (started by the job with the GGUF, -ngl, -t); raw /completion, prompt cache off."""
    import urllib.request
    prompt, _ = llm_prompt_fn(tok_path)

    def fn(e):
        body = json.dumps({"prompt": prompt(e), "n_predict": max_tokens, "temperature": 0, "cache_prompt": False}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/completion", body,
                                                                     {"Content-Type": "application/json"})).read())
        return parsed(r["content"]), r.get("tokens_predicted", 0)
    return {label: run_arm(fn, entries, label)}


def luna_arm(entries, label):
    """GPT-6 Luna, Responses API from the laptop; latency includes the network round trip. Key never printed."""
    import re
    from openai import OpenAI
    from arena.api_agents import env_key
    from actionman.schema import from_bfcl, to_openai_tool
    client = OpenAI(api_key=env_key("OPENAI_API_KEY"), timeout=60)

    def fn(e):
        tools, names = [], {}
        for t in from_bfcl(e, None, "").tools:
            f = to_openai_tool(t)["function"]
            names[re.sub(r"\.", "_", f["name"])] = f["name"]
            tools.append({"type": "function", **f, "name": re.sub(r"\.", "_", f["name"])})
        r = client.responses.create(model="gpt-6-luna", input=e["question"][0], tools=tools, store=False,
                                    reasoning={"effort": "low"})
        calls = []
        for it in r.output:
            if it.type == "function_call":
                try:
                    calls.append({names.get(it.name, it.name): json.loads(it.arguments or "{}")})
                except json.JSONDecodeError:
                    calls.append({names.get(it.name, it.name): {}})
        return calls, r.usage.output_tokens
    return {label: run_arm(fn, entries, label)}


def host_meta(threads):
    import platform
    import subprocess
    m = {"torch": torch.__version__, "threads": threads, "python": platform.python_version()}
    try:
        m["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except OSError:
        m["cpu"] = platform.processor()
    if torch.cuda.is_available():
        m["gpu"] = torch.cuda.get_device_name(0)
        try:
            m["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    for mod in ("vllm", "transformers"):
        try:
            m[mod] = __import__(mod).__version__
        except Exception:
            pass
    return m


def plan_main(args):
    ids = json.load(open(args.ids))["ids"][: args.n or None]
    by_id = {}
    for line in open(args.entries, encoding="utf-8"):
        e = json.loads(line)
        by_id[e["id"]] = e
    entries = [by_id[i] for i in ids]
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    res = {"meta": {**host_meta(args.threads or torch.get_num_threads()), "device": device, "n": len(entries),
                    "warm": WARM, "rep": args.rep}}
    for arm in args.arms.split(","):
        if arm in ("am_eager", "am_compiled"):
            res.update(am_arms(args.actionman, entries, device, args.T, arm == "am_compiled", args.mode))
        elif arm == "vllm":
            res.update(vllm_arm(args.llm, entries, args.label, args.max_num_seqs, dtype=args.dtype))
        elif arm == "hf":
            res.update(hf_arm(args.llm, entries, args.label, device))
        elif arm == "llamacpp":
            res.update(llamacpp_arm(args.llm, args.llama_url, entries, args.label))
        elif arm == "luna":
            res.update(luna_arm(entries, args.label))
        elif arm == "am_profile":
            res.update(am_profile(args.actionman, entries, args.mode))
    json.dump(res, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actionman", default=None)
    ap.add_argument("--qwen", default=None)
    ap.add_argument("--data", default="data/prepared/bfcl.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="latency.json")
    ap.add_argument("--compile", action="store_true", help="S6: also time torch.compile / CUDA-graph inference")
    # benchmark-plan mode (--ids): one engine per process; the job script runs arms in ABAB order
    ap.add_argument("--ids", default=None, help="bench/latency_ids.json (frozen BFCL v4 prompt ids)")
    ap.add_argument("--entries", default="data/bench/bfcl_v4_entries.jsonl")
    ap.add_argument("--arms", default="am_eager", help="am_eager,am_compiled,vllm,hf,llamacpp,luna,am_profile")
    ap.add_argument("--T", type=int, nargs="+", default=[3, 1])
    ap.add_argument("--mode", default="flat", choices=["flat", "last"], help="ActionMan prompt flattening")
    ap.add_argument("--llm", default=None, help="HF dir of the LLM (weights for vllm/hf, tokenizer for llamacpp)")
    ap.add_argument("--label", default="llm")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16", help="vllm arm weights/activations dtype")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rep", default="A")
    args = ap.parse_args()
    if args.ids:
        return plan_main(args)
    global COMPILE
    COMPILE = args.compile
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exs = [e for e in load_jsonl(args.data) if e.source != "bfcl_irrelevance"]
    step = max(1, len(exs) // args.n)
    exs = exs[::step][: args.n]
    res = {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    if args.actionman:
        res.update(bench_actionman(args.actionman, exs, device))
    if args.qwen:
        res.update(bench_qwen(args.qwen, exs, device))
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
