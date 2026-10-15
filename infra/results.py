"""Pull eval/latency JSONs from the private HF repo and write docs/results/results.md."""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from launch import REPO, ROOT, hf_token  # noqa: E402

CATS = [("xlam", "xLAM test"), ("xlam_neg", "xLAM no-tool"), ("bfcl_simple", "BFCL simple"),
        ("bfcl_multiple", "BFCL multiple"), ("bfcl_parallel", "BFCL parallel"),
        ("bfcl_parallel_multiple", "BFCL par-multiple"), ("bfcl_irrelevance", "BFCL irrelevance"), ("all", "All")]


def fetch(path):
    try:
        return json.load(open(hf_hub_download(REPO, path, token=hf_token(), force_download=True)))
    except Exception:
        return None


def main():
    files = HfApi(token=hf_token()).list_repo_files(REPO)
    cols = []  # (label, metrics-by-cat)
    for f in sorted(files):
        if f.endswith("/eval.json") or f.endswith("/eval_v3_compiled.json"):
            run = f.split("/")[1] + "/" + f.split("/")[2].replace("eval_v3_compiled.json", "compiled")
            for T, m in (fetch(f) or {}).items():
                cols.append((f"ActionMan {run} {T}", m))
        elif f.endswith("/eval_qwen.json"):
            cols.append((f"Qwen3-0.6B SFT {f.split('/')[1]}", fetch(f)))
    lat = [(f.split("/")[1], fetch(f)) for f in files if os.path.basename(f).startswith("latency") and f.endswith(".json")]
    lines = ["# ActionMan results (v1, v2, v3)", "", "Full-call accuracy (tool + every argument correct) / tool-name accuracy.", ""]
    lines.append("| | " + " | ".join(c for c, _ in cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for key, name in CATS:
        row = []
        for _, m in cols:
            v = (m or {}).get(key)
            row.append(f"{v['full_acc'] * 100:.1f} / {v['tool_acc'] * 100:.1f} (n={v['n']})" if v else "—")
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    metas = [(c, m.get("_meta")) for c, m in cols if m and m.get("_meta")]
    if metas:
        lines += ["", "| run | avg forward passes | ECE | eval seconds |", "|---|---|---|---|"]
        for c, mt in metas:
            if "avg_passes" not in mt:
                continue
            lines.append(f"| {c} | {mt.get('avg_passes', '—') if isinstance(mt.get('avg_passes'), str) else round(mt.get('avg_passes', 0), 2)} | "
                         f"{round(mt.get('ece', 0), 3)} | {round(mt.get('seconds', 0))} |")
    for run, L in lat:
        if not L:
            continue
        lines += ["", f"## Latency, batch 1, {L.get('device')} ({run})", "", "| model | p50 ms | p95 ms | mean ms | notes |", "|---|---|---|---|---|"]
        for k, v in L.items():
            if isinstance(v, dict):
                if "avg_forward_passes" in v:
                    note = f"{v['avg_forward_passes']:.2f} forward passes"
                elif "same_as_eager_pct" in v:
                    note = f"torch.compile + CUDA graphs; {v['same_as_eager_pct']:.1f}% outputs identical to eager"
                elif "avg_new_tokens" in v:
                    note = f"{v['avg_new_tokens']:.0f} generated tokens"
                else:
                    note = "CPU tokenisation, already included in ActionMan rows"
                lines.append(f"| {k} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} | {v['mean_ms']:.1f} | {note} |")
    out = os.path.join(ROOT, "docs", "results", "results.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
