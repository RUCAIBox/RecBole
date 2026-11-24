# Changelog

## [Unreleased] - 2026-06-03

### Compatibility fixes for Python 3.14 and updated dependencies

Resolved breakages introduced by migrating to Python 3.14 and upgrading
to NumPy 2.x, pandas 3.x, PyTorch 2.6+, scipy 1.14+, and hyperopt 0.2.7.

#### `recbole/config/configurator.py`
- Replaced removed NumPy 2.0 type aliases: `np.float_` → `np.float64`,
  `np.complex_` → `np.complex128`, `np.unicode_` → `np.str_`

#### `recbole/data/dataset/dataset.py`
- Fixed pandas 3.0 Copy-on-Write: `fillna(inplace=True)` on a column
  slice is now a no-op; replaced with explicit column reassignment (×3)
- Fixed pandas 3.0 `Series.agg(len)` semantic change: `agg` now
  aggregates the whole Series and returns a scalar instead of applying
  the function element-wise; replaced with `apply(len)` (×4)

#### `recbole/data/dataset/sequential_dataset.py`
- Same `agg(len)` → `apply(len)` fix for `item_list_length_field`
  computation, which was producing the total row count instead of
  per-sequence lengths

#### `recbole/data/dataloader/abstract_dataloader.py`
- Fixed PyTorch 2.x strict device-matching for advanced indexing: DNS
  negative sampling produced indices on the model device that were used
  to index a CPU tensor; added `.cpu()` to detach before indexing

#### `recbole/trainer/trainer.py`
- Replaced deprecated `import torch.cuda.amp as amp` with
  `import torch.amp as amp`
- Updated `amp.GradScaler(...)` → `amp.GradScaler(self.device.type, ...)`
  (×2) so the scaler always matches the actual compute device
- Updated `amp.autocast(...)` → `amp.autocast(self.device.type, ...)`
  so the autocast context matches the actual compute device
- Added `weights_only=False` to `torch.load` calls (×2): PyTorch 2.6
  changed the default to `weights_only=True`, which rejects RecBole
  checkpoints that contain non-tensor objects (config dicts, epoch
  state, etc.)

#### `recbole/quick_start/quick_start.py`
- Added `weights_only=False` to `torch.load` for the same reason

#### `recbole/trainer/hyper_tuning.py`
- Replaced `np.random.RandomState(seed)` with `np.random.default_rng(seed)`:
  hyperopt 0.2.7 calls `rng.integers()` which only exists on the new
  `numpy.random.Generator` interface, not on `RandomState`

#### Model files — `torch.load` (`weights_only=False`)
- `recbole/model/general_recommender/neumf.py`
- `recbole/model/general_recommender/nais.py`
- `recbole/model/general_recommender/convncf.py`
- `recbole/model/general_recommender/ract.py`
- `recbole/model/sequential_recommender/s3rec.py`
- `recbole/model/context_aware_recommender/kd_dagfm.py`

#### Model files — `dok_matrix._update` removal (scipy 1.14+)
- `recbole/model/general_recommender/lightgcn.py`
- `recbole/model/general_recommender/ngcf.py`
- `recbole/model/general_recommender/ncl.py`
- `recbole/model/general_recommender/gcmc.py`
- `recbole/model/general_recommender/spectralcf.py`
- `recbole/model/knowledge_aware_recommender/kgin.py`
- `recbole/model/knowledge_aware_recommender/mcclk.py`

  Replaced `A._update(data_dict)` with the public `A.update(data_dict)`;
  the private `_update` method was removed from `scipy.sparse.dok_matrix`

---

### Dependency updates

#### `requirements.txt`
- Pinned all packages to their currently installed versions
- Changed range specifiers (`>=`) to exact pins (`==`) for
  reproducibility
- Updated `ray==2.55.1` → `ray[tune]==2.55.1` to include the tune
  extras required by `recbole/quick_start/quick_start.py`
- Added new dependencies:
  - `lightgbm==4.6.0`
  - `xgboost==3.2.0`
  - `faiss-cpu==1.14.2`
  - `pytest==9.0.3`

---

### New files

#### `setup_extras.sh`
- Shell script for platform-specific setup that cannot be expressed in
  `requirements.txt`:
  - Installs `torch-scatter` from the PyG wheel index for
    PyTorch 2.12 + CUDA 13.0
  - Patches `dgl 0.1.3` (the only version available on PyPI) to use
    `collections.abc` imports instead of the bare `collections` aliases
    removed in Python 3.10+ (required for Python 3.14)
