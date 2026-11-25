#!/usr/bin/env python3
"""
Subsample users from a RecBole dataset directory to accelerate experiments.

This script creates a new dataset folder with the same RecBole TSV format,
filtering interactions and optional side files by a sampled subset of users.

Input expectation (standard RecBole layout):
  <input_dir>/<dataset_name>.inter  # required
  <input_dir>/<dataset_name>.user   # optional
  <input_dir>/<dataset_name>.item   # optional

Output layout:
  <output_dir>/<output_name>.inter
  <output_dir>/<output_name>.user   (if input exists)
  <output_dir>/<output_name>.item   (filtered or copied)

Example:
  python tools/subsample_users.py \
    --input_dir dataset/Amazon_Beauty \
    --dataset_name Amazon_Beauty \
    --sample_ratio 0.1 \
    --output_dir dataset \
    --output_name Amazon_Beauty_10p

Then train with:
  python run_recbole.py --model SASRec --dataset Amazon_Beauty_10p --config sasrec_baseline.yaml
"""

import argparse
import os
import random
import shutil
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _detect_prefix(input_dir: str, dataset_name: Optional[str]) -> Tuple[str, str]:
    """
    Detect dataset prefix path and dataset name.

    Returns (prefix_path, detected_dataset_name)
    where prefix_path = f"{input_dir}/{dataset_name}" without extension.
    """
    if dataset_name:
        prefix = os.path.join(input_dir, dataset_name)
        return prefix, dataset_name

    inter_candidates = [f for f in os.listdir(input_dir) if f.endswith('.inter')]
    if len(inter_candidates) == 1:
        dataset_name = inter_candidates[0][:-6]
        return os.path.join(input_dir, dataset_name), dataset_name

    base = os.path.basename(os.path.normpath(input_dir))
    expected = f"{base}.inter"
    if expected in inter_candidates:
        return os.path.join(input_dir, base), base

    raise ValueError(
        f"Cannot detect dataset name in '{input_dir}'. Provide --dataset_name explicitly. "
        f"Candidates: {inter_candidates}"
    )


def _read_header_and_indices(path: str, required_cols: List[str]) -> Tuple[str, str, Dict[str, int]]:
    """
    Read header line to determine delimiter and column indices for required columns.

    Returns (header_line, delimiter, {col_name: idx}).
    """
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
    if not header:
        raise ValueError(f"Empty file or missing header: {path}")

    delimiter = '\t' if '\t' in header else ' '
    cols = header.rstrip('\n').split(delimiter)
    name_to_index: Dict[str, int] = {}
    for i, col in enumerate(cols):
        base = col.split(':', 1)[0]
        if base in required_cols and base not in name_to_index:
            name_to_index[base] = i
    for req in required_cols:
        if req not in name_to_index:
            raise ValueError(f"Column '{req}' not found in header of {path}: {header.strip()}")
    return header, delimiter, name_to_index


def _iter_lines(path: str) -> Iterable[str]:
    with open(path, 'r', encoding='utf-8') as f:
        # skip header
        next(f)
        for line in f:
            if line and line != '\n':
                yield line


def _gather_user_counts(
    inter_path: str,
    delimiter: str,
    user_idx: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with open(inter_path, 'r', encoding='utf-8') as f:
        next(f)  # header
        for line in f:
            if not line or line == '\n':
                continue
            parts = line.rstrip('\n').split(delimiter)
            if len(parts) <= user_idx:
                continue
            uid = parts[user_idx]
            counts[uid] = counts.get(uid, 0) + 1
    return counts


def _sample_users(
    user_counts: Dict[str, int],
    ratio: float,
    seed: int,
    min_interactions: int,
) -> Set[str]:
    eligible = [u for u, c in user_counts.items() if c >= min_interactions]
    if not eligible:
        return set()
    sample_size = max(1, int(round(len(eligible) * ratio))) if ratio < 1.0 else len(eligible)
    rng = random.Random(seed)
    sampled = set(rng.sample(eligible, min(sample_size, len(eligible))))
    return sampled


def _filter_interactions(
    inter_in: str,
    inter_out: str,
    delimiter: str,
    user_idx: int,
    item_idx: int,
    sampled_users: Set[str],
) -> Set[str]:
    used_items: Set[str] = set()
    with open(inter_in, 'r', encoding='utf-8') as fin, open(inter_out, 'w', encoding='utf-8') as fout:
        header = fin.readline()
        fout.write(header)
        for line in fin:
            if not line or line == '\n':
                continue
            parts = line.rstrip('\n').split(delimiter)
            if len(parts) <= max(user_idx, item_idx):
                continue
            if parts[user_idx] in sampled_users:
                fout.write(line)
                used_items.add(parts[item_idx])
    return used_items


def _filter_user_file(
    user_in: str,
    user_out: str,
    sampled_users: Set[str],
) -> None:
    with open(user_in, 'r', encoding='utf-8') as fin, open(user_out, 'w', encoding='utf-8') as fout:
        header = fin.readline()
        fout.write(header)
        delimiter = '\t' if '\t' in header else ' '
        cols = header.rstrip('\n').split(delimiter)
        try:
            uid_idx = next(i for i, c in enumerate(cols) if c.split(':', 1)[0] == 'user_id')
        except StopIteration:
            raise ValueError(f"user_id column not found in {user_in}")
        for line in fin:
            if not line or line == '\n':
                continue
            parts = line.rstrip('\n').split(delimiter)
            if len(parts) <= uid_idx:
                continue
            if parts[uid_idx] in sampled_users:
                fout.write(line)


def _filter_item_file(
    item_in: str,
    item_out: str,
    used_items: Optional[Set[str]],
) -> None:
    if used_items is None:
        shutil.copyfile(item_in, item_out)
        return
    with open(item_in, 'r', encoding='utf-8') as fin, open(item_out, 'w', encoding='utf-8') as fout:
        header = fin.readline()
        fout.write(header)
        delimiter = '\t' if '\t' in header else ' '
        cols = header.rstrip('\n').split(delimiter)
        try:
            iid_idx = next(i for i, c in enumerate(cols) if c.split(':', 1)[0] == 'item_id')
        except StopIteration:
            raise ValueError(f"item_id column not found in {item_in}")
        for line in fin:
            if not line or line == '\n':
                continue
            parts = line.rstrip('\n').split(delimiter)
            if len(parts) <= iid_idx:
                continue
            if parts[iid_idx] in used_items:
                fout.write(line)


def _copy_other_files(input_dir: str, output_dir: str, known_basenames: Set[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(input_dir):
        src = os.path.join(input_dir, name)
        if not os.path.isfile(src):
            continue
        base = os.path.basename(name)
        # skip known files (we already re-generated them)
        if base in known_basenames:
            continue
        shutil.copyfile(src, os.path.join(output_dir, base))


def main() -> None:
    parser = argparse.ArgumentParser(description='Subsample users from a RecBole dataset folder.')
    parser.add_argument('--input_dir', required=True, help='Directory containing the source dataset files.')
    parser.add_argument('--dataset_name', default=None, help='Dataset name (prefix). If omitted, auto-detect from .inter file.')
    parser.add_argument('--output_dir', default='dataset', help='Directory to place the output dataset folder.')
    parser.add_argument('--output_name', default=None, help='Output dataset name (folder and file prefix).')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Fraction of users to sample (0-1].')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for user sampling.')
    parser.add_argument('--min_interactions', type=int, default=1, help='Minimum interactions per user to be eligible for sampling.')
    parser.add_argument('--item_filter', choices=['used', 'all'], default='used', help='Whether to filter items to those used in sampled interactions.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output folder if it exists.')

    args = parser.parse_args()

    if not (0 < args.sample_ratio <= 1.0):
        raise ValueError('--sample_ratio must be in (0, 1].')
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    prefix_in, detected_name = _detect_prefix(args.input_dir, args.dataset_name)
    dataset_name_in = detected_name

    inter_in = f"{prefix_in}.inter"
    user_in = f"{prefix_in}.user"
    item_in = f"{prefix_in}.item"

    if not os.path.isfile(inter_in):
        raise FileNotFoundError(f"Missing .inter file: {inter_in}")

    output_name = args.output_name or f"{dataset_name_in}_sub{int(round(args.sample_ratio * 100))}p"
    output_dir = os.path.join(args.output_dir, output_name)
    if os.path.exists(output_dir):
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    inter_out = os.path.join(output_dir, f"{output_name}.inter")
    user_out = os.path.join(output_dir, f"{output_name}.user")
    item_out = os.path.join(output_dir, f"{output_name}.item")

    # Determine column indices in .inter
    inter_header, inter_delim, inter_idx = _read_header_and_indices(inter_in, ['user_id', 'item_id'])
    user_idx = inter_idx['user_id']
    item_idx = inter_idx['item_id']

    # Pass 1: gather user counts
    user_counts = _gather_user_counts(inter_in, inter_delim, user_idx)
    total_users = len(user_counts)

    sampled_users = _sample_users(user_counts, args.sample_ratio, args.seed, args.min_interactions)
    if not sampled_users:
        raise RuntimeError('No users sampled. Try lowering --min_interactions or increasing --sample_ratio.')

    # Pass 2: write filtered interactions and collect used items
    used_items = _filter_interactions(inter_in, inter_out, inter_delim, user_idx, item_idx, sampled_users)

    # Optional: filter .user if exists
    if os.path.isfile(user_in):
        _filter_user_file(user_in, user_out, sampled_users)

    # Optional: filter/copy .item if exists
    if os.path.isfile(item_in):
        _filter_item_file(item_in, item_out, used_items if args.item_filter == 'used' else None)

    # Copy all other files in the input directory as-is (e.g., .kg, .link), but
    # avoid duplicating the regenerated .inter/.user/.item in the output directory.
    known = {os.path.basename(inter_in), os.path.basename(user_in), os.path.basename(item_in)}
    _copy_other_files(args.input_dir, output_dir, known_basenames=known)

    kept_users = len(sampled_users)
    kept_items = len(used_items)
    print('Done.')
    print(f"Input dataset: {dataset_name_in} @ {args.input_dir}")
    print(f"Users total: {total_users} -> kept: {kept_users} (~{kept_users / max(total_users, 1):.2%})")
    if os.path.isfile(item_in):
        print(f"Items kept: {kept_items} (item_filter={args.item_filter})")
    print(f"Output dataset: {output_name} @ {output_dir}")
    print(f"Files written: {os.path.basename(inter_out)}"
          f"{', ' + os.path.basename(user_out) if os.path.isfile(user_in) else ''}"
          f"{', ' + os.path.basename(item_out) if os.path.isfile(item_in) else ''}")


if __name__ == '__main__':
    main()


