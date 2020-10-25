# RecBole (伯乐)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)


[Docs] | [Datasets] |

[Docs]: https://github.com/RUCAIBox/RecBole
[Datasets]: https://github.com/RUCAIBox/RecDatasets


RecBole is developed based on Python and PyTorch for reproducing and developing recommendation algorithms in a unified,
comprehensive and efficient framework for research purpose.
Our library includes 52 recommendation algorithms, covering four major categories

    - General Recommendation
    - Sequential Recommendation
    - Context-aware Recommendation
    - Knowledge-based Recommendation

<p align="center">
  <img src="http://data.dgl.ai/asset/image/DGL-Arch.png" alt="RecBole v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: RecBole Overall Architecture
</p>

## Installation
To install this package:
```
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```

To verify this package has been successfully installed:
```
python run_test_example.py
```
