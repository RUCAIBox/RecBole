# Project code: individual recommenders (Task 1.1 / 1.2)

Trains the baselines, neighbourhood models and matrix factorisation with RecBole on ML-100K.
Every model is written to disk in the same format, so hybrid, evaluation and reranking code
can work from files without re-running RecBole.

## Setup

```bash
cd project
pip install -r requirements.txt
```

Use `project/requirements.txt`, not the one in the repository root. The root file pins `ray<=2.6.3`,
which cannot be installed on recent Python versions, and our code needs neither ray nor hyperopt.
The scripts import RecBole from this repository, not from a pip-installed `recbole`.
Tested with Python 3.11, PyTorch 2.14, NumPy 1.26 and 2.4, and pandas 3.0.

## Running

```bash
python run_model.py all                    # train + export every model in configs/models/
python run_model.py UserKNN ItemKNN BPR    # or just some of them
python run_model.py UserKNN --set k 20     # one-off override, e.g. for a quick experiment
python run_model.py BPR --no-tuned         # use the untuned defaults

python tune.py UserKNN ItemKNN BPR         # grid search on the validation set (see below)
```

Rough CPU run times: KNN models and Pop take a few seconds each, and BPR takes about 10 s.
Tuning takes 20 s (UserKNN), about 1 min (ItemKNN) and about 5 min (BPR).

## Models

| Experiment    | RecBole model | What it is |
|---------------|---------------|------------|
| `Random`      | Random        | naive baseline |
| `Pop`         | TrainPop (ours, `models.py`) | naive baseline: most-interacted items in the training set |
| `UserKNN`     | ItemKNN, `knn_method: user` | user-based neighbourhood CF, cosine similarity |
| `ItemKNN`     | ItemKNN, `knn_method: item` | item-based neighbourhood CF, cosine similarity |
| `AsymUserKNN` | AsymKNN, `knn_method: user` | user-based CF with asymmetric cosine |
| `AsymItemKNN` | AsymKNN, `knn_method: item` | item-based CF with asymmetric cosine |
| `BPR`         | BPR           | matrix factorisation trained with a pairwise ranking loss |

To add a model, drop a YAML into `configs/models/` (it needs at least `model: <RecBole model name>`)
and optionally a grid in `hyper/`.

## Configuration

Configs are applied in this order; later files override earlier ones:

1. `configs/base.yaml`: shared by every model. It sets the dataset, a random 80/10/10 split per user
   (seed 2020), full ranking evaluation and NDCG@10 as the validation metric.
   **Don't change this after models have been run**, or they will no longer share a split.
   `run_model.py` warns you if that happens.
2. `configs/models/<experiment>.yaml`: the model and its default hyperparameters.
3. `configs/tuned/<experiment>.yaml`: written by `tune.py`, and used automatically when present.

Only ratings of **3 stars or more** are kept, and each one counts as a positive interaction
(`val_interval` in `base.yaml`). 1- and 2-star ratings (17% of ML-100K) are dropped before splitting,
so they appear in no split and are never treated as hits.

## Tuning

`tune.py <experiment>` tries every combination in `hyper/<experiment>.yaml`, trains on *train*, and
selects by NDCG@10 on *valid*. The test set is never used for selection; test scores are only
logged in the CSV for reference. It writes:

- `outputs/tuning/<experiment>.csv`: every combination tried, best first
- `configs/tuned/<experiment>.yaml`: the best parameters

## Outputs (`outputs/`)

| File | Contents |
|------|----------|
| `split/{train,valid,test}.tsv` | `user_id, item_id, rating, timestamp`, with original MovieLens ids. Identical for all models. |
| `<experiment>/scores.npz` | `scores` [943 users x 1574 items; items with only 1-2 star ratings are dropped], `user_ids`, `item_ids`. Raw model scores for **every** pair; seen items are not masked. |
| `<experiment>/recs_valid.tsv` | top-100 per user, `user_id, item_id, rank, score`, with training items removed |
| `<experiment>/recs_test.tsv` | top-100 per user, with training **and** validation items removed |
| `<experiment>/metrics.json` | RecBole's own valid/test metrics, **as a sanity check only**. The project asks us to report our own metrics. |
| `<experiment>/config.yaml` | the hyperparameters actually used |

Models are trained on *train* only. Use `recs_valid` / validation scores to fit anything
(e.g. hybrid weights), and `recs_test` for final evaluation.

### Loading outputs in your own code

```python
from common import load_split, load_scores, load_recs

train, valid, test = load_split()
S_user, user_ids, item_ids = load_scores("UserKNN")   # rows/cols are identical across models,
S_bpr, _, _ = load_scores("BPR")                      # so matrices can be combined directly
recs = load_recs("ItemKNN", "test")                   # DataFrame: user_id, item_id, rank, score
```

Scores are on different scales for different models (KNN: sums of similarities; BPR: dot
products; Pop: count / max count), so normalise them (e.g. per-user min-max or z-score) before
combining them in a *weighted / regression* hybrid.

## Mixed Hybrid (list mixing)

The Mixed Hybrid is a **separate** approach from the Logistic Regression weighted hybrid.
It does **not** combine raw scores. It interleaves already-ranked recommendation lists,
deduplicates per user, and keeps the first `K` unique items.

Preferred sources (different paradigms):

| Source | Paradigm |
|--------|----------|
| `ItemKNN` | neighbourhood collaborative filtering |
| `BPR` | matrix factorisation / personalized ranking |
| `ContentBased` | content similarity (`CB` model, run through `run_model.py`) |

### Strategy

Two mixers are implemented in `mixed_hybrid.py`:

1. **RoundRobin** — take rank 1 from each model, then rank 2, … skipping duplicates.
2. **Quota round-robin** — in each round take up to `quotas[model]` unseen items from that
   model's list, then repeat until `top_k` unique items are filled.

Example quotas `ItemKNN=2 BPR=1 ContentBased=1`:

```text
Round 1: ItemKNN, ItemKNN, BPR, ContentBased
Round 2: ItemKNN, ItemKNN, BPR, ContentBased
...
```

Deduplication is **per user**. If one source runs out of candidates, the others continue.

### Regenerate the inputs

Generated files under `outputs/` are gitignored and may be absent. Recreate them with:

```bash
python run_model.py ItemKNN BPR ContentBased
```

`configs/models/ContentBased.yaml` runs the teammates' `CB` model unchanged (its defaults from
`recbole/properties/model/CB.yaml`, with the ML-100K content fields `movie_title` and `class`)
through the same pipeline as the other models. It therefore uses the same split and the same
seen-item filtering (valid: train hidden; test: train + valid hidden). Rank is derived from the
CB similarity scores; its `filter_interacted` setting does not matter because the export masks
seen items itself.

CB downloads `distilbert-base-uncased` from Hugging Face on the first run. If that fails with
`CERTIFICATE_VERIFY_FAILED` (HTTPS interception by antivirus/proxy), download it once through the
Windows certificate store and then run offline:

```bash
pip install truststore
python -c "import truststore; truststore.inject_into_ssl(); from huggingface_hub import snapshot_download; snapshot_download('distilbert-base-uncased', allow_patterns=['*.json','*.txt','*.safetensors'])"
$env:HF_HUB_OFFLINE = "1"     # PowerShell; export HF_HUB_OFFLINE=1 on bash
```

### Tune on validation, evaluate once on test

Primary selection metric: **NDCG@10** (same as `configs/base.yaml`).

```bash
python tune_mixed_hybrid.py
python run_mixed_hybrid.py
```

`tune_mixed_hybrid.py` never reads the test set for selection. It writes:

- `configs/mixed_hybrid_best.yaml` — frozen best quotas
- `outputs/MixedHybrid/tuning_valid.csv` — all validation configurations
- `outputs/MixedHybrid/recs_valid.tsv`

`run_mixed_hybrid.py` loads that frozen config, builds test recommendations, reports standalone
vs Mixed Hybrid metrics on the **same** test split, and writes:

- `outputs/MixedHybrid/recs_test.tsv` (`user_id, item_id, rank, source, score`)
- `outputs/MixedHybrid/test_metrics.json`

### Results (seed 2020, ML-100K, ratings >= 3)

Validation (selection by NDCG@10, full table in `outputs/MixedHybrid/tuning_valid.csv`):

| Configuration | Recall@10 | MRR@10 | NDCG@10 |
|---|---|---|---|
| RoundRobin | 0.2030 | 0.3715 | 0.2076 |
| ItemKNN=5 BPR=3 CB=2 | 0.2179 | 0.3803 | 0.2223 |
| **ItemKNN=4 BPR=4 CB=2** | 0.2161 | 0.3798 | **0.2226** |
| ItemKNN=3 BPR=5 CB=2 | 0.2156 | 0.3815 | 0.2221 |
| ItemKNN=4 BPR=3 CB=3 | 0.2017 | 0.3769 | 0.2133 |
| ItemKNN=2 BPR=3 CB=5 | 0.1677 | 0.3694 | 0.1909 |

Test (frozen config, evaluated once):

| Model | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
|---|---|---|---|---|---|
| ItemKNN | 0.2266 | 0.3966 | 0.2389 | 0.6903 | 0.1524 |
| BPR | 0.2434 | 0.4277 | 0.2578 | 0.7190 | 0.1628 |
| ContentBased | 0.0280 | 0.0651 | 0.0291 | 0.1866 | 0.0223 |
| Mixed Hybrid (4:4:2) | 0.2168 | 0.3966 | 0.2356 | 0.6797 | 0.1468 |

The Mixed Hybrid does not beat BPR. On validation, every extra slot given to ContentBased lowers
NDCG@10, because CB alone is roughly 8x weaker than the collaborative models, so the slots it gets
mostly replace hits from ItemKNN/BPR. The best mix gives CB the smallest share in the grid.

### Tests

```bash
cd project
python -m unittest discover -s tests -v
```

## Notes on RecBole

- **The `Pop` experiment uses our own `TrainPop` (`models.py`), not RecBole's `Pop`.** RecBole's Pop
  counts an item at most once per training batch (`cnt[item] = cnt[item] + 1` ignores repeats), and it
  also counts randomly sampled negatives. Its "popularity" took only about 50 distinct values. TrainPop
  reads the true counts from the training interaction matrix and scores items the same way
  (count / max count). RecBole's `pop.py` itself is left unchanged.
- Project-specific models go in `models.py`: add the class to `PROJECT_MODELS` and refer to it by
  name in a `configs/models/*.yaml` file.
- Best-checkpoint reloading is done in `common.py` because RecBole's own reload fails on PyTorch >= 2.6.
- NumPy 2 support: `recbole/config/configurator.py` referred to aliases removed in NumPy 2.0
  (`np.float_`, `np.complex_`, `np.unicode_`), so importing RecBole crashed.
