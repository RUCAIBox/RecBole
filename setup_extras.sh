#!/usr/bin/env bash
# Installs platform-specific extras and patches incompatible packages.
set -euo pipefail

# ── torch-scatter (pre-built wheel for PyTorch 2.12 + CUDA 13.0) ────────────
echo "Installing torch-scatter..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.12.0+cu130.html

# ── Patch dgl 0.1.3: collections.abc imports removed in Python 3.10+ ────────
echo "Patching dgl collections imports..."
DGL_DIR="$(python -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("dgl").origin).parent)')"

sed -i 's/from collections import Mapping, Iterable/from collections.abc import Mapping, Iterable/' "$DGL_DIR/utils.py"
sed -i 's/from collections import Iterable/from collections.abc import Iterable/' "$DGL_DIR/batched_graph.py"
sed -i 's/from collections import MutableMapping, namedtuple/from collections.abc import MutableMapping\nfrom collections import namedtuple/' "$DGL_DIR/view.py"
sed -i 's/from collections import MutableMapping, namedtuple/from collections.abc import MutableMapping\nfrom collections import namedtuple/' "$DGL_DIR/frame.py"

echo "Verifying dgl import..."
python -c "import dgl; print('dgl', dgl.__version__, 'OK')"

echo "Done."
