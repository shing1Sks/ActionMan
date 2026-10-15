"""Export a trained checkpoint dir (model.pt, run.json, encoder_config.json) as a Hugging Face model folder:
model.safetensors, config.json, tokenizer, the model card and the inference code. Loaded by actionman.api.
python -m actionman.export_hf --ckpt runs/hf/runs/s16-v3-5epb/m --out hf_export --card docs/release/model_card.md"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch

VERSION = "0.1"
CODE = ["__init__.py", "api.py", "audit.py", "cached.py", "data.py", "features.py", "infer.py", "infer_v3.py", "model.py",
        "schema.py", "scoring.py", "span.py", "v3feat.py"]                 # api.py's import closure (tests/test_api.py)


def export(model, tok, flags: dict, out: str, card: str | None = None):
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).contiguous().clone() for k, v in model.state_dict().items()}   # clone: no shared storage
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt", "version": VERSION})
    json.dump({"version": VERSION, "base": "answerdotai/ModernBERT-large", "flags": flags,
               "encoder": model.mlm.config.to_dict()}, open(os.path.join(out, "config.json"), "w"), indent=1)
    tok.save_pretrained(out)
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(out, "code", "actionman"), exist_ok=True)
    for f in CODE:
        shutil.copy(os.path.join(here, f), os.path.join(out, "code", "actionman", f))
    if card:
        shutil.copy(card, os.path.join(out, "README.md"))
        lic = os.path.join(os.path.dirname(card), "LICENSE")
        if os.path.exists(lic):
            shutil.copy(lic, os.path.join(out, "LICENSE"))
        assets = os.path.join(os.path.dirname(card), "assets")
        if os.path.isdir(assets):
            shutil.copytree(assets, os.path.join(out, "assets"), dirs_exist_ok=True)


def main():
    from .evaluate import is_cached, load_model, uses_markers, uses_v3
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="hf_export")
    ap.add_argument("--card", default=None)
    args = ap.parse_args()
    model, tok = load_model(args.ckpt)
    flags = dict(cached=is_cached(args.ckpt), markers=uses_markers(args.ckpt), v3=uses_v3(args.ckpt))
    export(model, tok, flags, args.out, args.card)
    print("exported", args.out, sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.out, "*"))))


if __name__ == "__main__":
    main()
