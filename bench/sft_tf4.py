"""On the box: our Qwen3-0.6B SFT as a dir that transformers 4.x (vLLM 0.11 venv) loads correctly.
The SFT was saved by transformers 5: tokenizer_config.json (extra_special_tokens as a list) fails in 4.x, and config.json
keeps rope_theta under rope_parameters, which 4.x ignores (silently falling back to rope_theta=10000; Qwen3 uses 1e6).
The tokenizer is Qwen/Qwen3-0.6B's unchanged (baseline/qwen.py loads it from there), so its original files are used,
and config.json gets the top-level rope_theta back.

python bench/sft_tf4.py /workspace/ckpt/qwen /workspace/ckpt/qwen4
"""
import json
import os
import sys

from huggingface_hub import snapshot_download

src, dst = sys.argv[1], sys.argv[2]
base = snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.json", "*.txt"])
os.makedirs(dst, exist_ok=True)
for d, files in ((src, ("generation_config.json", "model.safetensors")),
                 (base, ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"))):
    for f in files:
        if not os.path.lexists(os.path.join(dst, f)):
            os.symlink(os.path.realpath(os.path.join(d, f)), os.path.join(dst, f))
cfg = json.load(open(os.path.join(src, "config.json")))
rp = cfg.get("rope_parameters") or {}
cfg.setdefault("rope_theta", rp.get("rope_theta", 1000000))
cfg.setdefault("rope_scaling", None if rp.get("rope_type", "default") == "default" else rp)
cfg.setdefault("torch_dtype", cfg.get("dtype", "float32"))
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
print("sft (tf4 tokenizer, rope_theta", cfg["rope_theta"], ") ->", dst, sorted(os.listdir(dst)), flush=True)
