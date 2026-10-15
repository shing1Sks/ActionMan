"""Train ActionMan. Writes heartbeat (for the box watchdog), progress, metrics, and final weights."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from . import beat
from .data import FeatureDataset, collate, load_jsonl, token_batches
from .evaluate import evaluate
from .features import load_tokenizer
from .model import ActionMan


def tiny_config(tok):
    from transformers import ModernBertConfig
    return ModernBertConfig(vocab_size=50368, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=128, max_position_embeddings=1024, global_attn_every_n_layers=1,
                            pad_token_id=tok.pad_token_id)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    ap.add_argument("--tokenizer", default="answerdotai/ModernBERT-large")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--dev-n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="optional state_dict to start from")
    ap.add_argument("--cached", action="store_true", help="block-causal cached variant (Stage 4)")
    ap.add_argument("--markers", action="store_true", help="v2: #i call markers in the slot block")
    ap.add_argument("--v3", action="store_true", help="v3: copy switch, option slots, list item slots")
    ap.add_argument("--v3-mask", choices=["clean", "legacy"], default="clean",
                    help="v3 masking: clean = v3.1 fix (default); legacy = the original v3 run")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less memory, ~30% slower)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.tokenizer)
    if args.tiny:
        model = ActionMan.from_encoder(tiny_config(tok), tok, v3=args.v3)
    else:
        model = ActionMan.from_encoder(args.encoder, tok, attn_impl="sdpa", v3=args.v3)
    if args.init:
        model.load_state_dict({k: v.float() for k, v in torch.load(args.init, map_location="cpu").items()}, strict=False)
    if args.grad_ckpt:
        model.mlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)

    train = load_jsonl(os.path.join(args.data, args.train_file))
    if args.max_train:
        train = train[: args.max_train]
    dev = load_jsonl(os.path.join(args.data, "dev.jsonl"))[: args.dev_n]
    ds = FeatureDataset(tok, train, cached=args.cached, markers=args.markers, v3=args.v3)
    lengths = [ds.length(i) for i in range(len(ds))]

    head_ids = {id(p) for p in model.head_params()}
    groups = [{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
              {"params": model.head_params(), "lr": args.head_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6)
    steps_per_epoch = len(token_batches(lengths, args.tokens_per_batch, random.Random(0)))
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    json.dump({**vars(args), "steps": total, "n_train": len(train), "device": device},
              open(os.path.join(args.out, "run.json"), "w"), indent=1)
    model.mlm.config.to_json_file(os.path.join(args.out, "encoder_config.json"))

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    step, t0, toks, epoch = 0, time.time(), 0, 0
    while step < total:
        batches = token_batches(lengths, args.tokens_per_batch, rng)
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=args.workers,
                                             collate_fn=lambda items: collate(items, tok, random.Random(rng.random()),
                                                                              cached=args.cached, v3_mask=args.v3_mask),
                                             persistent_workers=False)
        model.train()
        for batch in loader:
            if batch is None:
                continue
            batch = to_device(batch, device)
            with amp:
                loss, parts = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            toks += int(batch["a_attn"].sum()) + (int(batch["b_attn"].sum()) if batch.get("b_ids") is not None else 0)
            if step % 20 == 0 or step == 1:
                el = time.time() - t0
                rec = {"step": step, "total": total, "epoch": epoch, "loss": round(loss.item(), 4), **parts,
                       "tok_s": int(toks / el), "elapsed_min": round(el / 60, 2),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if device == "cuda" else 0}
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            open(os.path.join(args.out, "heartbeat"), "w").write(str(time.time())); beat()
            if step >= total:
                break
        epoch += 1
        m = evaluate(model, tok, dev, T=2, device=device, cached=args.cached, markers=args.markers, v3=args.v3)
        print("dev", json.dumps(m["all"]), m["_meta"], flush=True)
        json.dump(m, open(os.path.join(args.out, f"dev_epoch{epoch}.json"), "w"), indent=1)
        open(os.path.join(args.out, "heartbeat"), "w").write(str(time.time()))

    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    torch.save(sd, os.path.join(args.out, "model.pt"))
    json.dump({"done": True, "steps": step, "minutes": (time.time() - t0) / 60}, open(os.path.join(args.out, "DONE"), "w"))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
