![RecBole Logo](asset/logo.png)

--------------------------------------------------------------------------------

# RecBole (伯乐)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)


[HomePage] | [Docs] | [Datasets] |

[HomePage]: https://recbole.io/
[Docs]: https://recbole.io/docs/
[Datasets]: https://github.com/RUCAIBox/RecDatasets


RecBole is developed based on Python and PyTorch for reproducing and developing recommendation algorithms in a unified,
comprehensive and efficient framework for research purpose.
Our library includes 52 recommendation algorithms, covering four major categories:

+ General Recommendation
+ Sequential Recommendation
+ Context-aware Recommendation
+ Knowledge-based Recommendation

We design a unified and flexible data file format, and provide the support for 25 benchmark recommendation datasets.
A user can apply the provided script to process the original data copy, or simply download the processed datasets
by our team.


<p align="center">
  <img src="asset/framework.png" alt="RecBole v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: RecBole Overall Architecture
</p>


## Feature
+ *General and extensible data structure.* We design general and extensible data structures to unify the formatting and
usage of various recommendation datasets.

+ *Comprehensive benchmark models and datasets.* We implement 52 commonly used recommendation algorithms, and provide
the formatted copies of 27 recommendation datasets.

+ *Efficient GPU-accelerated execution.* We optimize the efficiency of our library with a number of improved techniques
oriented to the GPU environment.

+ *Extensive and standard evaluation protocols.* We support a series of widely adopted evaluation protocols or settings
for testing and comparing recommendation algorithms.

## RecBole News
**10/xx/2020**: We release the first version of RecBole **v0.1.0 release**.


## Installation
RecBole works with the following operating systems:

* Linux
* Windows 10
* macOS X

RecBole requires Python version 3.6 or later.

RecBole requires torch version 1.2.0 or later. If you want to use RecBole with GPU,
please ensure that CUDA or cudatoolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).

### Install from conda

```
conda install recbole
```

### Install from pip

```
pip install recbole
```

### Install from source
```
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```

## Quick-Start
With the source code, you can use the provided script for initial usage of our library:

```
python run_recbole.py
```

This script will run the BPR model on the ml-100k dataset.

Typically, this example takes less than one minute. We will obtain some output like:

```
INFO ml-100k
The number of users: 944
Average actions of users: 106.04453870625663
The number of items: 1683
Average actions of items: 59.45303210463734
The number of inters: 100000
The sparsity of the dataset: 93.70575143257098%

INFO Evaluation Settings:
Group by user_id
Ordering: {'strategy': 'shuffle'}
Splitting: {'strategy': 'by_ratio', 'ratios': [0.8, 0.1, 0.1]}
Negative Sampling: {'strategy': 'by', 'distribution': 'uniform', 'by': 1}

INFO BPRMF(
    (user_embedding): Embedding(944, 64)
    (item_embedding): Embedding(1683, 64)
    (loss): BPRLoss()
)
Trainable parameters: 168128

INFO epoch 0 training [time: 0.27s, train loss: 27.7231]
INFO epoch 0 evaluating [time: 0.12s, valid_score: 0.021900]
INFO valid result:
recall@10: 0.0073  mrr@10: 0.0219  ndcg@10: 0.0093  hit@10: 0.0795  precision@10: 0.0088

...

INFO epoch 16 training [time: 0.19s, train loss: 2.2169]
INFO epoch 16 evaluating [time: 0.08s, valid_score: 0.298600]
INFO valid result:
recall@10: 0.2049  mrr@10: 0.2986  ndcg@10: 0.1836  hit@10: 0.6684  precision@10: 0.1147

INFO Finished training, best eval result in epoch 5
INFO Loading model structure and parameters from saved/***.pth
INFO best valid result:
recall@10: 0.2077  mrr@10: 0.3329  ndcg@10: 0.1992  hit@10: 0.6738  precision@10: 0.1264
INFO test result:
recall@10: 0.2076  mrr@10: 0.3796  ndcg@10: 0.2203  hit@10: 0.6769  precision@10: 0.1404
```

If you want to change the parameters, such as ``learning_rate``, ``embedding_size``, just set the additional command
parameters as you need:

```
python run_recbole.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:

```
python run_recbole.py --model=[model_name]
```

## RecBole Major Releases
| Releases  | Date   | Features |
|-----------|--------|-------------------------|
| v0.1.0    | 10/xx/2020 |  Basic RecBole |


## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/RecBole/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs


## Cite
If you find RecBole useful for your research or development, please cite the following paper.

```
@article{recbole,
    title={RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
    author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xingyu Pan and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Zhen Wang and Xiaoling Wang and Ji-Rong Wen},
    year={2020},
    journal={arXiv preprint arXiv:}
}
```

## The Team
RecBole is developed and maintained by [RUC, BUPT, ECNU](https://www.recbole.io/about.html).

## License
RecBole uses MIT License.
