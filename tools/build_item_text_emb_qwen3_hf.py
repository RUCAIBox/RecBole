#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build item text embeddings using a HuggingFace Qwen3 model by mean-pooling last hidden states.

Input: mapping CSV exported by tools/export_internal_item_mapping.py with columns:
  - internal_item_id
  - item_token
  - title  (recommended; if missing, will use item_token)

Output: item_text_emb.qwen3.npy with shape [n_items, hidden_size],
  row 0 = zeros (PAD), rows 1.. aligned with internal ids.

Example:
  python tools/build_item_text_emb_qwen3_hf.py \
    --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --output dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
    --batch_size 16 --max_length 128 --dtype float16 \
    --prompt_template "[TITLE] {text}"

Notes:
- Run this on your GPU machine with the Qwen3/2 Instruct model.
- The prompt_template is a plain text prefix; we embed the entire prompt+title string directly
  via the encoder forward pass (no generation), then mean-pool hidden states.
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize

# Make local project importable when running from repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset, data_preparation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Qwen3 HF-based item text embeddings")
    p.add_argument("--mapping", required=True, help="CSV from export_internal_item_mapping.py")
    p.add_argument("--model_name_or_path", required=True, help="HF model id or local path (e.g., Qwen/Qwen2-7B-Instruct)")
    p.add_argument("--output", required=True, help="Output .npy path for embeddings")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", default=None, help="cuda device like cuda:0, or cpu; default: auto")
    p.add_argument("--device_map", default=None, help="set to 'auto' to shard across devices (HF accelerate)")
    p.add_argument(
        "--prompt_template",
        default="[TITLE] {text}",
        help="Template to wrap raw title into text for embedding. Use {text} as placeholder.",
    )
    p.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Wrap prompt using tokenizer.apply_chat_template like chat demo.",
    )
    p.add_argument(
        "--use_causal_lm",
        action="store_true",
        help="Load AutoModelForCausalLM instead of AutoModel; we still mean-pool hidden states.",
    )
    p.add_argument(
        "--placeholder_text",
        default="N/A",
        help="Fallback text when title/token is empty.",
    )
    p.add_argument(
        "--pad_placeholder_text",
        default="[PAD]",
        help="Placeholder for PAD row to avoid empty input during encoding.",
    )
    p.add_argument(
        "--project_dim",
        type=int,
        default=256,
        help="Reduce embedding dim via TruncatedSVD to this size, then L2-normalize (default: 256).",
    )
    p.add_argument(
        "--svd_random_state",
        type=int,
        default=42,
        help="Random state for TruncatedSVD when --project_dim is set.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Optional: RecBole dataset name (e.g., Amazon_Beauty) to derive train-only ids for SVD fitting.",
    )
    p.add_argument(
        "--config",
        nargs="+",
        default=[],
        help="Optional: YAML config files to customize dataset/model params when deriving train/valid/test splits.",
    )
    return p.parse_args()


def _select_device(dev: str | None) -> torch.device:
    if dev is not None:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    if getattr(tokenizer, "pad_token_id", None) is None:
        # No padding token available: encode one by one
        outs = []
        for txt in texts:
            enc = tokenizer(
                txt,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc, output_hidden_states=True, return_dict=True)
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                last_hidden = outputs.hidden_states[-1]
            attn_mask = enc.get("attention_mask", torch.ones_like(last_hidden[:, :, 0]))
            mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, dim=1)
            outs.append(emb.to(torch.float32).detach().cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    else:
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        # If tokenizer produced empty sequences (rare), fall back to per-item path
        if enc["input_ids"].shape[1] == 0:
            outs = []
            for txt in texts:
                enc_i = tokenizer(
                    txt,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc_i = {k: v.to(device) for k, v in enc_i.items()}
                outputs_i = model(**enc_i, output_hidden_states=True, return_dict=True)
                last_hidden_i = getattr(outputs_i, "last_hidden_state", None)
                if last_hidden_i is None:
                    last_hidden_i = outputs_i.hidden_states[-1]
                attn_mask_i = enc_i.get("attention_mask", torch.ones_like(last_hidden_i[:, :, 0]))
                mask_i = attn_mask_i.unsqueeze(-1).type_as(last_hidden_i)
                summed_i = (last_hidden_i * mask_i).sum(dim=1)
                counts_i = mask_i.sum(dim=1).clamp(min=1e-6)
                emb_i = summed_i / counts_i
                emb_i = torch.nn.functional.normalize(emb_i, dim=1)
                outs.append(emb_i.to(torch.float32).detach().cpu().numpy())
            return np.concatenate(outs, axis=0)

        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            last_hidden = outputs.hidden_states[-1]
        attn_mask = enc.get("attention_mask", torch.ones_like(last_hidden[:, :, 0]))  # [B, L]
        mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        emb = summed / counts
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.to(torch.float32).detach().cpu().numpy()


def main():
    args = parse_args()

    df = pd.read_csv(args.mapping)
    if "internal_item_id" not in df.columns or "item_token" not in df.columns:
        raise ValueError("mapping CSV must contain 'internal_item_id' and 'item_token'")
    has_title = "title" in df.columns

    device = _select_device(args.device)
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        # Prefer existing special tokens to avoid adding new tokens (some tokenizers forbid it)
        for cand in [getattr(tokenizer, "eos_token", None), getattr(tokenizer, "unk_token", None), getattr(tokenizer, "bos_token", None)]:
            if isinstance(cand, str) and tokenizer.convert_tokens_to_ids(cand) is not None:
                tokenizer.pad_token = cand
                break

    # Auto-detect Qwen config → prefer CausalLM class unless user forces otherwise
    try:
        hf_cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model_type = getattr(hf_cfg, "model_type", "") or hf_cfg.__class__.__name__
        is_qwen_like = "qwen" in str(model_type).lower()
    except Exception:
        hf_cfg = None
        is_qwen_like = False

    use_causal = args.use_causal_lm or is_qwen_like

    if use_causal:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            low_cpu_mem_usage=True,
        )
    if args.device_map is None:
        model = model.to(device)
    model.eval()

    # Build texts in internal id order (after tokenizer, to allow chat templates)
    df = df.sort_values("internal_item_id")
    texts: List[str] = []
    for _, row in df.iterrows():
        if row["internal_item_id"] == 0:
            raw = args.pad_placeholder_text
        else:
            raw = str(row["title"]) if has_title else str(row["item_token"])
            if len(raw.strip()) == 0:
                raw = args.placeholder_text
        base_prompt = args.prompt_template.replace("{text}", raw.strip())
        if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": base_prompt}]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(chat_text)
        else:
            texts.append(base_prompt)

    print(f"Encoding {len(texts)} items with batch_size={args.batch_size} max_length={args.max_length}...")
    all_emb = []
    with tqdm(total=len(texts), unit="items") as pbar:
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i : i + args.batch_size]
            emb = encode_batch(model, tokenizer, batch, args.max_length, device, torch_dtype)
            all_emb.append(emb)
            pbar.update(len(batch))
    mat = np.concatenate(all_emb, axis=0)

    # Ensure PAD row is zeros
    if mat.shape[0] > 0:
        mat[0, :] = 0.0

    # Optional dimensionality reduction to a target size
    if args.project_dim is not None:
        orig_dim = mat.shape[1]
        target_dim = int(args.project_dim)

        if target_dim <= 0:
            raise ValueError("--project_dim must be > 0")

        if target_dim == orig_dim:
            # Just L2-normalize (non-PAD rows)
            if mat.shape[0] > 1:
                mat[1:, :] = l2_normalize(mat[1:, :], norm="l2", axis=1, copy=False)
        elif target_dim < orig_dim:
            # Fit SVD on train-only rows to avoid leakage, then transform all non-PAD rows
            # Determine train internal ids if dataset/config provided
            train_ids = None
            if args.dataset is not None and len(args.dataset) > 0:
                try:
                    cfg = Config(model="BPR", dataset=args.dataset, config_file_list=args.config)
                    ds = create_dataset(cfg)
                    train_data, valid_data, test_data = data_preparation(cfg, ds)
                    iid_field = cfg["ITEM_ID_FIELD"]
                    train_ids_raw = train_data.dataset.inter_feat[iid_field].numpy()
                    train_ids = np.unique(train_ids_raw).astype(np.int64)
                    # Exclude PAD id 0
                    train_ids = train_ids[train_ids > 0]
                except Exception:
                    train_ids = None

            nonpad_all = mat[1:, :].astype(np.float32, copy=False)
            if train_ids is None or len(train_ids) == 0:
                train_subset = nonpad_all
            else:
                # Intersect with existing rows (mat is aligned by internal_item_id)
                max_row = mat.shape[0] - 1
                train_ids = train_ids[(train_ids >= 1) & (train_ids <= max_row)]
                if len(train_ids) == 0:
                    train_subset = nonpad_all
                else:
                    train_subset = mat[train_ids, :].astype(np.float32, copy=False)

            # Guard tiny feature dims
            svd_k = max(1, min(target_dim, train_subset.shape[1] - 1 if train_subset.shape[1] > 1 else 1))
            svd = TruncatedSVD(n_components=svd_k, random_state=args.svd_random_state)
            svd.fit(train_subset)
            reduced = svd.transform(nonpad_all)
            # If svd_k < target_dim, right-pad zeros
            if svd_k < target_dim:
                pad = np.zeros((reduced.shape[0], target_dim - svd_k), dtype=reduced.dtype)
                reduced = np.concatenate([reduced, pad], axis=1)
            reduced = l2_normalize(reduced, norm="l2", axis=1, copy=False)
            mat_proj = np.zeros((mat.shape[0], target_dim), dtype=np.float32)
            mat_proj[1:, :] = reduced
            mat = mat_proj
        else:  # target_dim > orig_dim → pad zeros
            mat_pad = np.zeros((mat.shape[0], target_dim), dtype=np.float32)
            mat_pad[:, :orig_dim] = mat
            if mat.shape[0] > 1:
                mat_pad[1:, :] = l2_normalize(mat_pad[1:, :], norm="l2", axis=1, copy=False)
            mat = mat_pad

    # Save
    if args.dtype == "float16":
        mat = mat.astype(np.float16)
    elif args.dtype == "bfloat16":
        # numpy doesn't have bfloat16 well-supported; keep float32 on disk
        mat = mat.astype(np.float32)
    else:
        mat = mat.astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, mat)
    print(
        f"Saved Qwen3 embeddings to: {os.path.abspath(args.output)}  "
        f"shape={mat.shape}  dtype={mat.dtype}  (original_dim={all_emb[0].shape[1] if len(all_emb)>0 else 'NA'}, project_dim={args.project_dim})"
    )


if __name__ == "__main__":
    main()


