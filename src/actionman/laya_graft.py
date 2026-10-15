"""Stage 1 test: does Laya's encoder still 'know words'? Graft base ModernBERT's MLM head onto Laya's encoder
and compare masked-LM loss with base ModernBERT on our own queries + tool signatures."""
from __future__ import annotations

import argparse
import json
import os
import random

import torch

from .data import load_jsonl
from .schema import signature


@torch.no_grad()
def mlm_loss(model, tok, texts, device, ratio=0.15, seed=0):
    rng = random.Random(seed)
    tot, n = 0.0, 0
    for t in texts:
        ids = tok(t, truncation=True, max_length=256, return_tensors="pt").input_ids[0]
        cand = list(range(1, len(ids) - 1))
        if not cand:
            continue
        pos = rng.sample(cand, max(1, int(ratio * len(cand))))
        labels = torch.full_like(ids, -100)
        labels[pos] = ids[pos]
        inp = ids.clone(); inp[pos] = tok.mask_token_id
        out = model(input_ids=inp[None].to(device), labels=labels[None].to(device))
        tot += out.loss.item() * len(pos); n += len(pos)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/prepared/dev.jsonl")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, ModernBertForMaskedLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    exs = load_jsonl(args.data)[: args.n]
    texts = [e.query for e in exs] + [signature(t) for e in exs[:100] for t in e.tools[:1]]

    base = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-large").to(device).eval()
    res = {"base_modernbert": mlm_loss(base, tok, texts, device)}
    laya_sd = load_file(hf_hub_download("convaiinnovations/laya", "model.safetensors"))
    enc = {k[len("encoder."):]: v.float() for k, v in laya_sd.items() if k.startswith("encoder.")}
    missing, unexpected = base.model.load_state_dict(enc, strict=False)
    res["laya_encoder_missing_keys"] = len(missing)
    res["laya_encoder_unexpected_keys"] = len(unexpected)
    base.tie_weights()
    res["laya_encoder_plus_base_head"] = mlm_loss(base, tok, texts, device)
    res["ratio"] = res["laya_encoder_plus_base_head"] / res["base_modernbert"]
    res["verdict"] = "start from Laya" if res["ratio"] <= 1.05 else "start from base ModernBERT"
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "laya_graft.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
