"""Qwen3-0.6B baseline: full SFT on the same prepared data, Hermes-style <tool_call> JSON output.

python baseline/qwen.py train   --data data/prepared --out runs/qwen
python baseline/qwen.py eval    --ckpt runs/qwen --data data/prepared/test.jsonl data/prepared/bfcl.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from actionman import beat  # noqa: E402
from actionman.data import load_jsonl, token_batches  # noqa: E402
from actionman.schema import Call, to_openai_tool     # noqa: E402
from actionman.scoring import match_calls, names_match  # noqa: E402

NO_CALL = "None of the available tools can do this."


def prompt_text(tok, ex) -> str:
    return tok.apply_chat_template([{"role": "user", "content": ex.query}], tools=[to_openai_tool(t) for t in ex.tools],
                                   add_generation_prompt=True, enable_thinking=False, tokenize=False)


def target_text(ex) -> str:
    if not ex.calls:
        return NO_CALL + "<|im_end|>"
    return "\n".join("<tool_call>\n" + json.dumps({"name": c.name, "arguments": c.args}, ensure_ascii=False)
                     + "\n</tool_call>" for c in ex.calls) + "<|im_end|>"


def parse_calls(text: str) -> list[Call]:
    out = []
    for m in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            d = json.loads(m)
            out.append(Call(d.get("name", ""), d.get("arguments", {}) or {}))
        except Exception:
            out.append(Call("__unparseable__", {}))
    return out


def encode(tok, ex, max_len=2048):
    p = tok(prompt_text(tok, ex), add_special_tokens=False).input_ids
    t = tok(target_text(ex), add_special_tokens=False).input_ids
    if len(p) + len(t) > max_len:
        return None
    return p + t, [-100] * len(p) + t


def load(ckpt_or_name, tiny=False, device="cpu", train=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config
    tok = AutoTokenizer.from_pretrained(ckpt_or_name if os.path.isdir(ckpt_or_name) else "Qwen/Qwen3-0.6B")
    if tiny:
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=4096)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt_or_name, torch_dtype=torch.bfloat16 if (device == "cuda" and not train) else torch.float32,
                                                     attn_implementation="sdpa")
    return model.to(device), tok


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load("Qwen/Qwen3-0.6B", args.tiny, device, train=True)   # fp32 master weights + bf16 autocast
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    exs = load_jsonl(os.path.join(args.data, args.train_file))
    if args.max_train:
        exs = exs[: args.max_train]
    enc = [e for e in (encode(tok, ex) for ex in exs) if e is not None]
    rng = random.Random(0)
    lengths = [len(x) for x, _ in enc]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    per_epoch = len(token_batches(lengths, args.tokens_per_batch, random.Random(0)))
    total = args.max_steps or int(per_epoch * args.epochs)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    step, t0, toks = 0, time.time(), 0
    model.train()
    while step < total:
        for idx in token_batches(lengths, args.tokens_per_batch, rng):
            L = max(lengths[i] for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(idx), L), -100, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                x, y = enc[i]
                ids[r, :len(x)] = torch.tensor(x); lab[r, :len(y)] = torch.tensor(y); att[r, :len(x)] = 1
            ids, lab, att = ids.to(device), lab.to(device), att.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1; toks += int(att.sum())
            if step % 20 == 0 or step == 1:
                el = time.time() - t0
                rec = {"step": step, "total": total, "loss": round(loss.item(), 4), "tok_s": int(toks / el), "elapsed_min": round(el / 60, 2)}
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            open(os.path.join(args.out, "heartbeat"), "w").write(str(time.time())); beat()
            if step >= total:
                break
    model.config.use_cache = True
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    json.dump({"done": True, "steps": step, "minutes": (time.time() - t0) / 60}, open(os.path.join(args.out, "DONE"), "w"))


@torch.no_grad()
def generate(model, tok, exs, device, batch=16, max_new=256):
    tok.padding_side = "left"
    outs = []
    for s in range(0, len(exs), batch):
        texts = [prompt_text(tok, ex) for ex in exs[s:s + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        beat()
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=False)
    return outs


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.ckpt, False, device) if not args.tiny else load("Qwen/Qwen3-0.6B", True, device)
    model.eval()
    exs = [e for p in args.data for e in load_jsonl(p)]
    if args.limit:
        exs = exs[: args.limit]
    t0 = time.time()
    texts = generate(model, tok, exs, device)
    stats = collections.defaultdict(collections.Counter)
    for ex, text in zip(exs, texts):
        calls = parse_calls(text)
        for key in (ex.source, "all"):
            s = stats[key]; s["n"] += 1; s["full"] += match_calls(calls, ex.gold); s["names"] += names_match(calls, ex.gold)
    res = {k: {"n": s["n"], "full_acc": s["full"] / s["n"], "tool_acc": s["names"] / s["n"]} for k, s in stats.items()}
    res["_meta"] = {"seconds": time.time() - t0}
    out = args.out or os.path.join(args.ckpt if not args.tiny else ".", "eval_qwen.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: (round(v["full_acc"], 3), round(v["tool_acc"], 3)) for k, v in res.items() if k != "_meta"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--data", nargs="+", default=["data/prepared"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "train":
        args.data = args.data[0]
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
