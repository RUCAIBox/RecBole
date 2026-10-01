"""Shared helpers: build configs, train a RecBole model, export its outputs and load them back.

Everything downstream (hybrids, evaluation metrics, rerankers) should read the
files written here instead of re-running RecBole, so all parts of the project
work on exactly the same split and the same model scores.

Output layout (under project/outputs/):
    split/train.tsv, valid.tsv, test.tsv   user_id, item_id, rating, timestamp (original MovieLens ids)
    <experiment>/scores.npz                 full score matrix [n_users x n_items] + user_ids / item_ids
    <experiment>/recs_valid.tsv             top-N per user, training items removed
    <experiment>/recs_test.tsv              top-N per user, training + validation items removed
    <experiment>/metrics.json               RecBole's own valid/test metrics (sanity check only)
    <experiment>/config.yaml                the hyperparameters that were used
"""

import hashlib
import json
import os
import sys
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)  # use this repo's RecBole, not a pip-installed one

from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.data.interaction import Interaction  # noqa: E402
from recbole.utils import get_trainer, init_logger, init_seed  # noqa: E402

from models import PROJECT_MODELS  # noqa: E402

# RecBole's NaN filling triggers this pandas>=3 warning; ML-100K interactions have no NaNs, so it is harmless.
warnings.filterwarnings("ignore", message="A value is being set on a copy of a DataFrame")

CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
SPLIT_DIR = os.path.join(OUTPUT_DIR, "split")


# --------------------------------------------------------------------------- #
# Config / training
# --------------------------------------------------------------------------- #
def experiment_names():
    """All experiments that have a config in configs/models/."""
    return sorted(f[:-5] for f in os.listdir(os.path.join(CONFIG_DIR, "models")) if f.endswith(".yaml"))


def config_files(experiment, use_tuned=True):
    """base.yaml -> models/<experiment>.yaml -> tuned/<experiment>.yaml (later files win)."""
    model_file = os.path.join(CONFIG_DIR, "models", f"{experiment}.yaml")
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"No config for experiment '{experiment}'. Available: {', '.join(experiment_names())}"
        )
    files = [os.path.join(CONFIG_DIR, "base.yaml"), model_file]
    tuned_file = os.path.join(CONFIG_DIR, "tuned", f"{experiment}.yaml")
    if use_tuned and os.path.exists(tuned_file):
        files.append(tuned_file)
    return files


def build_config(experiment, overrides=None, use_tuned=True):
    overrides = dict(overrides or {})
    overrides.setdefault("data_path", os.path.join(REPO_DIR, "dataset"))
    overrides.setdefault("checkpoint_dir", os.path.join(OUTPUT_DIR, "_checkpoints"))
    # RecBole also reads `--key=value` from sys.argv with the highest priority, which would
    # clash with our own CLI flags, so hide the command line while the config is built.
    files = config_files(experiment, use_tuned)
    with open(files[1]) as f:
        model_name = yaml.safe_load(f)["model"]
    # Models from models.py are passed as classes; RecBole's own models by name.
    model = PROJECT_MODELS.get(model_name, model_name)
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        return Config(model=model, config_file_list=files, config_dict=overrides)
    finally:
        sys.argv = argv


def train(experiment, overrides=None, use_tuned=True, verbose=True):
    """Train one experiment. Returns (config, dataset, model, train_data, valid_data, test_data, result)."""
    config = build_config(experiment, overrides, use_tuned)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    if verbose:
        logger.info(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = config.model_class(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    # Restore the best (early-stopped) weights ourselves: RecBole's own reload uses torch.load's
    # default weights_only=True since PyTorch 2.6, which cannot read RecBole checkpoints.
    checkpoint = torch.load(trainer.saved_model_file, map_location=config["device"], weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    os.remove(trainer.saved_model_file)
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

    result = {
        "best_valid_score": float(best_valid_score),
        "valid": {k: float(v) for k, v in best_valid_result.items()},
        "test": {k: float(v) for k, v in test_result.items()},
    }
    if verbose:
        logger.info(f"[{experiment}] valid: {result['valid']}")
        logger.info(f"[{experiment}] test : {result['test']}")
    return config, dataset, model, train_data, valid_data, test_data, result


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_matrix(model, dataset, batch_size=256):
    """Scores of every (user, item) pair as a numpy array indexed by RecBole's internal ids.

    Row/column 0 is RecBole's padding id and is set to -inf.
    """
    model.eval()
    n_users, n_items = dataset.user_num, dataset.item_num
    scores = np.full((n_users, n_items), -np.inf, dtype=np.float32)
    for start in range(1, n_users, batch_size):
        users = torch.arange(start, min(start + batch_size, n_users))
        inter = Interaction({dataset.uid_field: users}).to(model.device)
        batch = model.full_sort_predict(inter).view(len(users), -1)
        scores[start:start + len(users)] = batch.float().cpu().numpy()
    scores[:, 0] = -np.inf
    return scores


def _to_frame(split_dataset, dataset):
    """Interactions of a split, with RecBole's internal ids mapped back to MovieLens ids."""
    inter = split_dataset.inter_feat
    uid, iid = dataset.uid_field, dataset.iid_field
    frame = pd.DataFrame({
        "user_id": dataset.id2token(uid, inter[uid].numpy()),
        "item_id": dataset.id2token(iid, inter[iid].numpy()),
    })
    for col in ("rating", "timestamp"):
        if col in inter.columns:
            frame[col] = inter[col].numpy()
    return frame


def _digest(frames):
    h = hashlib.md5()
    for f in frames:
        h.update(pd.util.hash_pandas_object(f, index=False).values.tobytes())
    return h.hexdigest()


def export_split(dataset, train_data, valid_data, test_data):
    """Write the split once; on later runs check the new run produced the identical split."""
    frames = [_to_frame(d._dataset, dataset) for d in (train_data, valid_data, test_data)]
    digest = _digest(frames)
    digest_file = os.path.join(SPLIT_DIR, "split.md5")
    logger = getLogger()
    if os.path.exists(digest_file):
        with open(digest_file) as f:
            if f.read().strip() != digest:
                logger.warning(
                    "The split of this run differs from outputs/split/. Did you change base.yaml? "
                    "Delete outputs/ and re-run ALL models so they share one split."
                )
        return
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for name, frame in zip(("train", "valid", "test"), frames):
        frame.to_csv(os.path.join(SPLIT_DIR, f"{name}.tsv"), sep="\t", index=False)
    with open(digest_file, "w") as f:
        f.write(digest)
    logger.info(f"Split written to {SPLIT_DIR}")


def _history_mask(*split_datasets):
    """Boolean [n_users x n_items] mask of the interactions in the given splits."""
    mask = None
    for d in split_datasets:
        m = d._dataset.inter_matrix(form="csr").astype(bool)
        mask = m if mask is None else (mask + m)
    return mask.toarray()


def _top_n(scores, mask, n, dataset):
    """Top-n items per user after masking out already-seen items."""
    masked = scores.copy()
    masked[mask] = -np.inf
    n = min(n, masked.shape[1] - 1)
    top = np.argpartition(-masked[1:], n, axis=1)[:, :n]  # skip padding user 0
    top_scores = np.take_along_axis(masked[1:], top, axis=1)
    order = np.argsort(-top_scores, axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)

    users = np.repeat(np.arange(1, masked.shape[0]), n)
    return pd.DataFrame({
        "user_id": dataset.id2token(dataset.uid_field, users),
        "item_id": dataset.id2token(dataset.iid_field, top.ravel()),
        "rank": np.tile(np.arange(1, n + 1), masked.shape[0] - 1),
        "score": top_scores.ravel(),
    })


def export(experiment, config, dataset, model, train_data, valid_data, test_data, result, top_n=100):
    out_dir = os.path.join(OUTPUT_DIR, experiment)
    os.makedirs(out_dir, exist_ok=True)
    export_split(dataset, train_data, valid_data, test_data)

    scores = score_matrix(model, dataset)
    np.savez_compressed(
        os.path.join(out_dir, "scores.npz"),
        scores=scores[1:, 1:],  # drop padding row/column
        user_ids=np.array(dataset.id2token(dataset.uid_field, np.arange(1, dataset.user_num))),
        item_ids=np.array(dataset.id2token(dataset.iid_field, np.arange(1, dataset.item_num))),
    )

    # Validation recs: model has seen train only -> hide train items.
    # Test recs: hide train + validation items (same protocol as RecBole's full-sort evaluation).
    _top_n(scores, _history_mask(train_data), top_n, dataset).to_csv(
        os.path.join(out_dir, "recs_valid.tsv"), sep="\t", index=False
    )
    _top_n(scores, _history_mask(train_data, valid_data), top_n, dataset).to_csv(
        os.path.join(out_dir, "recs_test.tsv"), sep="\t", index=False
    )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    # Record the model's own hyperparameters: everything set in its model/tuned config files.
    keys = {"seed"}
    for file in config_files(experiment)[1:]:
        with open(file) as f:
            keys.update(yaml.safe_load(f) or {})
    params = {k: config[k] for k in sorted(keys) if k in config and isinstance(config[k], (int, float, str, dict))}
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    getLogger().info(f"[{experiment}] outputs written to {out_dir}")


# --------------------------------------------------------------------------- #
# Loading (use these from hybrid / evaluation / reranking code)
# --------------------------------------------------------------------------- #
def load_split():
    """Returns (train, valid, test) DataFrames with columns user_id, item_id, rating, timestamp."""
    read = lambda name: pd.read_csv(  # noqa: E731
        os.path.join(SPLIT_DIR, f"{name}.tsv"), sep="\t", dtype={"user_id": str, "item_id": str}
    )
    return read("train"), read("valid"), read("test")


def load_scores(experiment):
    """Returns (scores, user_ids, item_ids).

    scores[u, i] is the model's score for user user_ids[u] and item item_ids[i].
    user_ids / item_ids are the same for every experiment, so matrices can be combined directly.
    Already-seen items are NOT masked here.
    """
    data = np.load(os.path.join(OUTPUT_DIR, experiment, "scores.npz"))
    return data["scores"], data["user_ids"], data["item_ids"]


def load_recs(experiment, stage="test"):
    """Top-N lists (user_id, item_id, rank, score). stage is 'valid' or 'test'."""
    return pd.read_csv(
        os.path.join(OUTPUT_DIR, experiment, f"recs_{stage}.tsv"),
        sep="\t", dtype={"user_id": str, "item_id": str},
    )
