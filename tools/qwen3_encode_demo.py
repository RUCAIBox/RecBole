#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal Qwen3 encoding demo to sanity-check tokenization, padding, and embedding shapes.

Usage:
  python tools/qwen3_encode_demo.py \
    --model /home/charlie/project/qwen/Model \
    --texts "Hello world" "测试一下" "[TITLE] N/A" \
    --use_causal_lm --use_chat_template --max_length 64 --dtype float16 --device_map auto
"""

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--texts", nargs="+", default=["Hello", "测试", "N/A"]) 
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", default=None)
    p.add_argument("--device_map", default=None)
    p.add_argument("--use_causal_lm", action="store_true")
    p.add_argument("--use_chat_template", action="store_true")
    return p.parse_args()


def select_device(dev):
    return torch.device(dev) if dev else torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode(model, tokenizer, texts, max_length, device):
    # Build chat template if requested
    processed = []
    for t in texts:
        base = t.strip() or "N/A"
        if hasattr(tokenizer, "apply_chat_template"):
            msg = [{"role": "user", "content": base}]
            s = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        else:
            s = base
        processed.append(s)

    # Encode with safe padding fallback
    if getattr(tokenizer, "pad_token_id", None) is None:
        outs = []
        for s in processed:
            enc = tokenizer(s, padding=False, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, return_dict=True)
            last = getattr(out, "last_hidden_state", None) or out.hidden_states[-1]
            mask = enc.get("attention_mask", torch.ones_like(last[:, :, 0])).unsqueeze(-1).type_as(last)
            emb = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            emb = torch.nn.functional.normalize(emb, dim=1)
            outs.append(emb.to(torch.float32).cpu().numpy())
        return np.concatenate(outs, axis=0)
    else:
        enc = tokenizer(processed, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, return_dict=True)
        last = getattr(out, "last_hidden_state", None) or out.hidden_states[-1]
        mask = enc.get("attention_mask", torch.ones_like(last[:, :, 0])).unsqueeze(-1).type_as(last)
        emb = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.to(torch.float32).cpu().numpy()


def main():
    args = parse_args()
    device = select_device(args.device)
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None and isinstance(getattr(tok, "eos_token", None), str):
        tok.pad_token = tok.eos_token

    if args.use_causal_lm:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch_dtype, device_map=args.device_map, low_cpu_mem_usage=True
        )
    else:
        model = AutoModel.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch_dtype, device_map=args.device_map, low_cpu_mem_usage=True
        )
    if args.device_map is None:
        model = model.to(device)
    model.eval()

    emb = encode(model, tok, args.texts, args.max_length, device)
    print("Embeddings shape:", emb.shape)
    print("First row (norm):", np.linalg.norm(emb[0]))


if __name__ == "__main__":
    main()


