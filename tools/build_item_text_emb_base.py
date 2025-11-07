#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build base item text embeddings (TF-IDF + TruncatedSVD) aligned with RecBole internal item indices.

Outputs a matrix `item_text_emb.npy` with shape [n_items, d_text], where row 0 is [PAD] (all zeros),
and rows 1..n_items-1 align to RecBole internal item ids. This file can be consumed by RLMRec and
other models that accept `item_text_emb_path`.

Design goals:
- Fair base (no LLM): character-level TF-IDF over item titles, then SVD to a fixed dimension.
- Deterministic and reproducible: controlled random_state, frozen vectors.
- No information leakage: only uses static item features.

Usage example:
  python tools/build_item_text_emb_base.py \
    --dataset Amazon_Beauty \
    --config recbole/properties/model/GRU4RecCPR.yaml \
    --output data/Amazon_Beauty/item_text_emb.base.npy \
    --title_field title \
    --svd_dim 256

Notes:
- The script reads the raw `.item` file via RecBole dataset metadata to get titles as plain text.
- If the specified title field is missing, it will try common fallbacks and then fall back to empty strings.
- For Chinese and multilingual text, char-level n-grams work robustly without additional tokenizers.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize

# Make local project importable when running from repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset, data_preparation


def _detect_item_file(dataset) -> Optional[str]:
    """Return path to `<dataset_name>.item` if exists, else None."""
    dataset_dir = getattr(dataset, "dataset_path", None)
    dataset_name = getattr(dataset, "dataset_name", None)
    if not dataset_dir or not dataset_name:
        return None
    candidate = os.path.join(dataset_dir, f"{dataset_name}.item")
    return candidate if os.path.exists(candidate) else None


def _find_col_by_base(df: pd.DataFrame, base_names: List[str]) -> Optional[str]:
    """Find a column whose base name (before ':') matches one of base_names."""
    cols = list(df.columns)
    # exact match first
    for name in base_names:
        if name in cols:
            return name
    # typed header like `field:token`
    for name in base_names:
        for c in cols:
            if isinstance(c, str) and c.split(":")[0] == name:
                return c
    return None


def _choose_title_field(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    """Pick a reasonable title/textual field from `.item` DataFrame.

    Priority: preferred -> common candidates; supports typed headers like `title:token`.
    """
    if preferred:
        col = _find_col_by_base(df, [preferred])
        if col is not None:
            return col
    candidates = [
        "title",
        "item_title",
        "name",
        "item_name",
        "product_title",
        "product_name",
        "categories",
        "category",
    ]
    return _find_col_by_base(df, candidates)


def _build_token_to_title_map(
    item_df: pd.DataFrame, item_id_col: str, title_col: Optional[str]
) -> Dict[str, str]:
    """Map external item tokens -> raw title string (empty string if missing)."""
    token_to_title: Dict[str, str] = {}
    if title_col is None:
        # No textual column; map to empty string
        for tok in item_df[item_id_col].astype(str).tolist():
            token_to_title[tok] = ""
        return token_to_title

    # Ensure strings and NaNs handled
    titles = item_df[title_col].fillna("")
    # If title is not string (e.g., numbers), cast to string
    titles = titles.astype(str)
    for tok, title in zip(item_df[item_id_col].astype(str).tolist(), titles.tolist()):
        token_to_title[tok] = title.strip()
    return token_to_title


def _get_internal_item_tokens(dataset) -> List[str]:
    """Get internal id-ordered external tokens for items, including PAD at 0."""
    iid_field = dataset.iid_field
    n_items = dataset.num(iid_field)
    ids = np.arange(n_items, dtype=np.int64)
    tokens = dataset.id2token(iid_field, ids)
    # Ensure list of str
    return [str(t) for t in tokens.tolist()]


def _build_texts_in_internal_order(
    internal_tokens: List[str], token_to_title: Dict[str, str]
) -> List[str]:
    texts: List[str] = []
    for tok in internal_tokens:
        if tok == "[PAD]":
            texts.append("")
        else:
            texts.append(token_to_title.get(tok, ""))
    return texts


def _fit_tfidf_svd(
    texts: List[str],
    n_components: int,
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: Optional[int] = None,
    random_state: int = 42,
) -> np.ndarray:
    """Compute TF-IDF then reduce with TruncatedSVD, then L2 normalize rows.

    Returns dense array of shape [len(texts), n_components].
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        norm=None,  # we'll normalize after SVD
        dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform(texts)

    # Handle edge cases where vocabulary is tiny
    svd_k = max(1, min(n_components, tfidf.shape[1] - 1 if tfidf.shape[1] > 1 else 1))
    svd = TruncatedSVD(n_components=svd_k, random_state=random_state)
    reduced = svd.fit_transform(tfidf)

    # If reduced dim < requested, pad zeros to target dim
    if svd_k < n_components:
        pad = np.zeros((reduced.shape[0], n_components - svd_k), dtype=reduced.dtype)
        reduced = np.concatenate([reduced, pad], axis=1)

    # L2 normalize; keep PAD row (index 0) as zeros afterwards
    reduced = l2_normalize(reduced, norm="l2", axis=1, copy=False)
    reduced[0, :] = 0.0
    return reduced.astype(np.float32)


def _fit_on_train_transform_all(
    train_texts: List[str],
    all_texts: List[str],
    n_components: int,
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: Optional[int] = None,
    random_state: int = 42,
) -> np.ndarray:
    """Fit TF-IDF and SVD on train_texts only, then transform all_texts.

    Returns dense array of shape [len(all_texts), n_components].
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        norm=None,
        dtype=np.float32,
    )
    # Fit only on training texts
    tfidf_train = vectorizer.fit_transform(train_texts)
    tfidf_all = vectorizer.transform(all_texts)

    # Handle edge cases where vocabulary is tiny
    svd_k = max(1, min(n_components, tfidf_train.shape[1] - 1 if tfidf_train.shape[1] > 1 else 1))
    svd = TruncatedSVD(n_components=svd_k, random_state=random_state)
    svd.fit(tfidf_train)
    reduced = svd.transform(tfidf_all)

    # If reduced dim < requested, pad zeros to target dim
    if svd_k < n_components:
        pad = np.zeros((reduced.shape[0], n_components - svd_k), dtype=reduced.dtype)
        reduced = np.concatenate([reduced, pad], axis=1)

    # L2 normalize; keep PAD row (index 0) as zeros afterwards
    reduced = l2_normalize(reduced, norm="l2", axis=1, copy=False)
    reduced[0, :] = 0.0
    return reduced.astype(np.float32)


def build_item_text_emb(
    dataset_name: str,
    config_files: List[str],
    output_path: str,
    title_field: Optional[str] = None,
    svd_dim: int = 256,
    analyzer: str = "char",
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    max_features: Optional[int] = None,
    dtype: str = "float16",
) -> str:
    """Main pipeline to build base item text embeddings and save to output_path.

    Returns the absolute output path.
    """
    if not dataset_name:
        raise KeyError("--dataset is required (e.g., --dataset Amazon_Beauty)")
    cfg = Config(model="BPR", dataset=dataset_name, config_file_list=config_files)
    dataset = create_dataset(cfg)

    item_file = _detect_item_file(dataset)
    if item_file is None:
        print("[WARN] .item file not found; falling back to empty titles for all items.")
        item_df = pd.DataFrame({cfg["ITEM_ID_FIELD"]: []})
    else:
        item_df = pd.read_csv(item_file, sep="\t")

    item_id_col = _find_col_by_base(item_df, [cfg["ITEM_ID_FIELD"], "item_id", "item", "iid"]) or cfg["ITEM_ID_FIELD"]

    chosen_title = _choose_title_field(item_df, title_field)
    token_to_title = _build_token_to_title_map(item_df, item_id_col, chosen_title)
    internal_tokens = _get_internal_item_tokens(dataset)
    texts_all = _build_texts_in_internal_order(internal_tokens, token_to_title)

    # Build split and collect train-only item ids to avoid leakage
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    iid_field = cfg["ITEM_ID_FIELD"]
    try:
        train_iids = train_data.dataset.inter_feat[iid_field].numpy()
    except Exception:
        # Fallback: no split info available → treat all non-PAD as train (kept for robustness)
        train_iids = np.arange(1, len(internal_tokens), dtype=np.int64)
    train_iids_set = set([int(x) for x in np.unique(train_iids).tolist() if int(x) > 0])
    # Assemble train texts by internal id index (exclude PAD=0)
    train_texts = [texts_all[i] for i in range(1, len(texts_all)) if i in train_iids_set]
    # Guard: if train_texts ends up empty, fall back to all (rare/corrupt case)
    if len(train_texts) == 0:
        print("[WARN] train_texts is empty; falling back to fitting on all_texts (may risk leakage).")
        emb = _fit_tfidf_svd(
            texts_all,
            n_components=svd_dim,
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
        )
    else:
        emb = _fit_on_train_transform_all(
            train_texts=train_texts,
            all_texts=texts_all,
            n_components=svd_dim,
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
        )

    # Cast dtype if requested
    if dtype == "float16":
        emb = emb.astype(np.float16)
    elif dtype == "float32":
        emb = emb.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, emb)
    print(f"Saved item_text_emb to: {os.path.abspath(output_path)}  shape={emb.shape}  dtype={emb.dtype}")
    return os.path.abspath(output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build base item text embeddings (TF-IDF+SVD)")
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g., Amazon_Beauty",
    )
    p.add_argument(
        "--config",
        nargs="+",
        required=False,
        default=[],
        help="Optional YAML config files to customize dataset/model params",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for the generated .npy file (e.g., data/Amazon_Beauty/item_text_emb.base.npy)",
    )
    p.add_argument(
        "--title_field",
        default=None,
        help="Column name in .item for title; if omitted, tries common names like 'title'",
    )
    p.add_argument("--svd_dim", type=int, default=256, help="Output embedding dimension")
    p.add_argument(
        "--ngram_min", type=int, default=1, help="Minimum n for character n-grams"
    )
    p.add_argument(
        "--ngram_max", type=int, default=2, help="Maximum n for character n-grams"
    )
    p.add_argument(
        "--min_df", type=int, default=2, help="Min document frequency for TF-IDF vocabulary"
    )
    p.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Limit TF-IDF vocabulary size (None means unlimited)",
    )
    p.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Output dtype for the saved matrix",
    )
    return p.parse_args()


def main():
    args = parse_args()
    build_item_text_emb(
        dataset_name=args.dataset,
        config_files=args.config,
        output_path=args.output,
        title_field=args.title_field,
        svd_dim=args.svd_dim,
        analyzer="char",
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=args.max_features,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()


