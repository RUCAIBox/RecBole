![RecBole Logo](asset/logo.png)

--------------------------------------------------------------------------------

# RecBole (伯乐)

*“世有伯乐，然后有千里马。千里马常有，而伯乐不常有。”——韩愈《马说》*

[![PyPi Latest Release](https://img.shields.io/pypi/v/recbole)](https://pypi.org/project/recbole/)
[![Conda Latest Release](https://anaconda.org/aibox/recbole/badges/version.svg)](https://anaconda.org/aibox/recbole)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-RecBole-%23B21B1B)](https://arxiv.org/abs/2011.01731)


[HomePage] | [Docs] | [Datasets] | [Paper] | [Blogs] | [Models] | [中文版]

[HomePage]: https://recbole.io/
[Docs]: https://recbole.io/docs/
[Datasets]: https://github.com/RUCAIBox/RecDatasets
[Paper]: https://arxiv.org/abs/2011.01731
[Blogs]: https://blog.csdn.net/Turinger_2000/article/details/111182852
[Models]: https://github.com/RUCAIBox/RecBole2.0/blob/main/model_list.md
[中文版]: README_CN.md

RecBole is developed based on Python and PyTorch for reproducing and developing recommendation algorithms in a unified,
comprehensive and efficient framework for research purpose.
Our library includes 78 recommendation algorithms, covering four major categories:

+ General Recommendation
+ Sequential Recommendation
+ Context-aware Recommendation
+ Knowledge-based Recommendation

We design a unified and flexible data file format, and provide the support for 28 benchmark recommendation datasets.
A user can apply the provided script to process the original data copy, or simply download the processed datasets
by our team.


<p align="center">
  <img src="asset/framework.png" alt="RecBole v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: RecBole Overall Architecture
</p>

In order to support the study of recent advances in recommender systems, we construct an extended recommendation library [RecBole2.0](https://github.com/RUCAIBox/RecBole2.0) consisting of 8 packages for up-to-date topics and architectures (e.g., debiased, fairness and GNNs). 

## Feature
+ **General and extensible data structure.** We design general and extensible data structures to unify the formatting and
usage of various recommendation datasets.

+ **Comprehensive benchmark models and datasets.** We implement 78 commonly used recommendation algorithms, and provide
the formatted copies of 28 recommendation datasets.

+ **Efficient GPU-accelerated execution.** We optimize the efficiency of our library with a number of improved techniques
oriented to the GPU environment.

+ **Extensive and standard evaluation protocols.** We support a series of widely adopted evaluation protocols or settings
for testing and comparing recommendation algorithms.


## RecBole News
![new](/asset/new.gif) **11/06/2022**: We release [the optimal hyperparameters of the model and their tuning ranges](https://recbole.io/hyperparameters/index.html).

![new](/asset/new.gif) **10/05/2022**: We release RecBole [v1.1.1](https://github.com/RUCAIBox/RecBole/releases/tag/v1.1.1).

![new](/asset/new.gif) **06/28/2022**: We release [**RecBole2.0**](https://github.com/RUCAIBox/RecBole2.0) with **8 packages** consisting of **65 newly implement models**. 

**02/25/2022**: We release RecBole [v1.0.1](https://github.com/RUCAIBox/RecBole/releases/tag/v1.0.1).

**09/17/2021**: We release RecBole [v1.0.0](https://github.com/RUCAIBox/RecBole/releases/tag/v1.0.0).

**03/22/2021**: We release RecBole [v0.2.1](https://github.com/RUCAIBox/RecBole/releases/tag/v0.2.1).

**01/15/2021**: We release RecBole [v0.2.0](https://github.com/RUCAIBox/RecBole/releases/tag/v0.2.0).

**12/10/2020**: 我们发布了[RecBole小白入门系列中文博客（持续更新中）](https://blog.csdn.net/Turinger_2000/article/details/111182852) 。

**12/06/2020**: We release RecBole [v0.1.2](https://github.com/RUCAIBox/RecBole/releases/tag/v0.1.2).

**11/29/2020**: We constructed preliminary experiments to test the time and memory cost on three
different-sized datasets and provided the [test result](https://github.com/RUCAIBox/RecBole#time-and-memory-costs)
for reference.

**11/03/2020**: We release the first version of RecBole **v0.1.1**.

### Latest Update for SIGIR 2023 Submission

To better meet the user requirements and contribute to the research community, we present a significant update of RecBole in the latest version, making it more user-friendly and easy-to-use as a comprehensive benchmark library for recommendation. We summarize these updates in "**Towards a More User-Friendly and Easy-to-Use Benchmark Library for Recommender Systems**" and submit the paper to **SIGIR 2023**. The main contribution in this update is introduced below.

Our extensions are made in three major aspects, namely the models/datasets, the framework, and the configurations. Furthermore, we provide more comprehensive documentation and well-organized FAQ for the usage of our library, which largely improves the user experience. More specifically, the highlights of this update are summarized as: 

1. We introduce more operations and settings to help benchmarking the recommendation domain.

2. We improve the user friendliness of our library by providing more detailed documentation and well-organized frequently asked questions. 
3. We point out several development guidelines for the open-source library developers. 

These extensions make it much easier to reproduce the benchmark results and stay up-to-date with the recent advances on recommender systems. The datailed comparison between this update and previous versions is listed below.

|          Aspect           |            RecBole 1.0             |          RecBole 2.0           |                   This update                    |
| :-----------------------: | :--------------------------------: | :----------------------------: | :----------------------------------------------: |
|   Recommendation tasks    |            4 categories            |    3 topics and 5 packages     |                   4 categories                   |
|    Models and datasets    |     73 models and 28 datasets      |  65 models and 8 new datasets  |            86 models and 41 datasets             |
|      Data structure       | Implemented Dataset and Dataloader |         Task-oriented          |  Compatible data module inherited from PyTorch   |
|    Continuous features    |          Field embedding           |        Field embedding         |        Field embedding and discretization        |
| GPU-accelerated execution |       Single-GPU utilization       |     Single-GPU utilization     |      Multi-GPU and mixed precision training      |
|  Hyper-parameter tuning   |       Serial gradient search       |     Serial gradient search     | Three search methods in both serial and parallel |
|     Significance test     |                 -                  |               -                |               Available interface                |
|     Benchmark results     |                 -                  | Partially public (GNN and CDR) |      Benchmark configurations on 82 models       |
|      Friendly usage       |           Documentation            |         Documentation          |       Improved documentation and FAQ page        |


## Installation
RecBole works with the following operating systems:

* Linux
* Windows 10
* macOS X

RecBole requires Python version 3.7 or later.

RecBole requires torch version 1.7.0 or later. If you want to use RecBole with GPU,
please ensure that CUDA or cudatoolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).

### Install from conda

```bash
conda install -c aibox recbole
```

### Install from pip

```bash
pip install recbole
```

### Install from source
```bash
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```

## Quick-Start
With the source code, you can use the provided script for initial usage of our library:

```bash
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
Negative Sampling: {'strategy': 'full', 'distribution': 'uniform'}
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
INFO epoch 63 training [time: 0.19s, train loss: 4.7660]
INFO epoch 63 evaluating [time: 0.08s, valid_score: 0.394500]
INFO valid result:
recall@10: 0.2156  mrr@10: 0.3945  ndcg@10: 0.2332  hit@10: 0.7593  precision@10: 0.1591
INFO Finished training, best eval result in epoch 52
INFO Loading model structure and parameters from saved/***.pth
INFO best valid result:
recall@10: 0.2169  mrr@10: 0.4005  ndcg@10: 0.235  hit@10: 0.7582  precision@10: 0.1598
INFO test result:
recall@10: 0.2368  mrr@10: 0.4519  ndcg@10: 0.2768  hit@10: 0.7614  precision@10: 0.1901
```

If you want to change the parameters, such as ``learning_rate``, ``embedding_size``, just set the additional command
parameters as you need:

```bash
python run_recbole.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:

```bash
python run_recbole.py --model=[model_name]
```

### Auto-tuning Hyperparameter 
Open `RecBole/hyper.test` and set several hyperparameters to auto-searching in parameter list. The following has two ways to search best hyperparameter:
* **loguniform**: indicates that the parameters obey the uniform distribution, randomly taking values from e^{-8} to e^{0}.
* **choice**: indicates that the parameter takes discrete values from the setting list.

Here is an example for `hyper.test`: 
```
learning_rate loguniform -8, 0
embedding_size choice [64, 96 , 128]
train_batch_size choice [512, 1024, 2048]
mlp_hidden_size choice ['[64, 64, 64]','[128, 128]']
```
Set training command parameters as you need to run:
```
python run_hyper.py --model=[model_name] --dataset=[data_name] --config_files=xxxx.yaml --params_file=hyper.test
e.g.
python run_hyper.py --model=BPR --dataset=ml-100k --config_files=test.yaml --params_file=hyper.test
```
Note that `--config_files=test.yaml` is optional, if you don't have any customize config settings, this parameter can be empty.

This processing maybe take a long time to output best hyperparameter and result:
```
running parameters:                                                                                                                    
{'embedding_size': 64, 'learning_rate': 0.005947474154838498, 'mlp_hidden_size': '[64,64,64]', 'train_batch_size': 512}                
  0%|                                                                                           | 0/18 [00:00<?, ?trial/s, best loss=?]
```

More information about parameter tuning can be found in our [docs](https://recbole.io/docs/user_guide/usage/parameter_tuning.html).


## Time and Memory Costs
We constructed preliminary experiments to test the time and memory cost on three different-sized datasets 
(small, medium and large). For detailed information, you can click the following links.

* [General recommendation models](asset/time_test_result/General_recommendation.md)
* [Sequential recommendation models](asset/time_test_result/Sequential_recommendation.md)
* [Context-aware recommendation models](asset/time_test_result/Context-aware_recommendation.md)
* [Knowledge-based recommendation models](asset/time_test_result/Knowledge-based_recommendation.md)

NOTE: Our test results only gave the approximate time and memory cost of our implementations in the RecBole library
(based on our machine server).  Any feedback or suggestions about the implementations and test are welcome. 
We will keep improving our implementations, and update these test results.


## RecBole Major Releases
| Releases | Date       |
|----------|------------|
| v1.1.1   | 10/05/2022 |
| v1.0.0   | 09/17/2021 |
| v0.2.0   | 01/15/2021 |
| v0.1.1   | 11/03/2020 |


## Open Source Contributions
As a one-stop framework from data processing, model development, algorithm training to scientific evaluation, RecBole has a total of **11** related GitHub projects including 
- two versions of RecBole ([RecBole 1.0](https://github.com/RUCAIBox/RecBole) and [RecBole 2.0](https://github.com/RUCAIBox/RecBole2.0));
- 8 benchmarking packages ([RecBole-MetaRec](https://github.com/nuster1128/RecBole-MetaRec), [RecBole-DA](https://github.com/RUCAIBox/RecBole-DA), [RecBole-Debias](https://github.com/JingsenZhang/RecBole-Debias), [RecBole-FairRec](https://github.com/TangJiakai/RecBole-FairRec), [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR), [RecBole-TRM](https://github.com/RUCAIBox/RecBole-TRM), [RecBole-GNN](https://github.com/RUCAIBox/RecBole-GNN) and [RecBole-PJF](https://github.com/RUCAIBox/RecBole-PJF));
- dataset repository (<a href="https://github.com/RUCAIBox/RecSysDatasets">RecSysDatasets</a>).

In the following table, we summarize the open source contributions of GitHub projects based on RecBole.

| **Projects**                                                 | **Stars**                                                    | **Forks**                                                    | **Issues**                                                   | **Pull requests**                                            |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [**RecBole**](https://github.com/RUCAIBox/RecBole)           | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole?style=social&logo=github)](https://github.com/RUCAIBox/RecBole/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole?style=social&logo=git)](https://github.com/RUCAIBox/RecBole/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole/pulls) |
| [**RecBole2.0**](https://github.com/RUCAIBox/RecBole2.0)     | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole2.0?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole2.0/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole2.0?style=social&logo=github)](https://github.com/RUCAIBox/RecBole2.0/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole2.0?style=social&logo=git)](https://github.com/RUCAIBox/RecBole2.0/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole2.0?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole2.0/pulls) |
| [**RecBole-DA**](https://github.com/RUCAIBox/RecBole-DA)     | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole-DA?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole-DA/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-DA?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-DA/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-DA?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-DA/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-DA?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-DA/pulls) |
| [**RecBole-MetaRec**](https://github.com/nuster1128/RecBole-MetaRec) | [![Stars](https://img.shields.io/github/stars/nuster1128/RecBole-MetaRec?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/nuster1128/RecBole-MetaRec/stargazers) | [![Forks](https://img.shields.io/github/forks/nuster1128/RecBole-MetaRec?style=social&logo=github)](https://github.com/nuster1128/RecBole-MetaRec/network/members) | [![Issues](https://img.shields.io/github/issues-closed/nuster1128/RecBole-MetaRec?style=social&logo=git)](https://github.com/nuster1128/RecBole-MetaRec/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/nuster1128/RecBole-MetaRec?style=social&logo=githubactions)](https://github.com/nuster1128/RecBole-MetaRec/pulls) |
| [**RecBole-Debias**](https://github.com/JingsenZhang/RecBole-Debias) | [![Stars](https://img.shields.io/github/stars/JingsenZhang/RecBole-Debias?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/JingsenZhang/RecBole-Debias/stargazers) | [![Forks](https://img.shields.io/github/forks/JingsenZhang/RecBole-Debias?style=social&logo=github)](https://github.com/JingsenZhang/RecBole-Debias/network/members) | [![Issues](https://img.shields.io/github/issues-closed/JingsenZhang/RecBole-Debias?style=social&logo=git)](https://github.com/JingsenZhang/RecBole-Debias/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/JingsenZhang/RecBole-Debias?style=social&logo=githubactions)](https://github.com/JingsenZhang/RecBole-Debias/pulls) |
| [**RecBole-FairRec**](https://github.com/TangJiakai/RecBole-FairRec) | [![Stars](https://img.shields.io/github/stars/TangJiakai/RecBole-FairRec?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/TangJiakai/RecBole-FairRec/stargazers) | [![Forks](https://img.shields.io/github/forks/TangJiakai/RecBole-FairRec?style=social&logo=github)](https://github.com/TangJiakai/RecBole-FairRec/network/members) | [![Issues](https://img.shields.io/github/issues-closed/TangJiakai/RecBole-FairRec?style=social&logo=git)](https://github.com/TangJiakai/RecBole-FairRec/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/TangJiakai/RecBole-FairRec?style=social&logo=githubactions)](https://github.com/TangJiakai/RecBole-FairRec/pulls) |
| [**RecBole-CDR**](https://github.com/RUCAIBox/RecBole-CDR)   | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole-CDR?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole-CDR/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-CDR?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-CDR/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-CDR?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-CDR/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-CDR?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-CDR/pulls) |
| [**RecBole-GNN**](https://github.com/RUCAIBox/RecBole-GNN)   | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole-GNN?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole-GNN/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-GNN?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-GNN/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-GNN?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-GNN/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-GNN?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-GNN/pulls) |
| [**RecBole-TRM**](https://github.com/RUCAIBox/RecBole-TRM)   | [![Stars](https://img.shields.io/github/stars/RUCAIBOX/RecBole-TRM?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBOX/RecBole-TRM/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-TRM?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-TRM/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-TRM?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-TRM/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-TRM?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-TRM/pulls) |
| [**RecBole-PJF**](https://github.com/RUCAIBox/RecBole-PJF)   | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole-PJF?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole-PJF/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-PJF?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-PJF/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-PJF?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-PJF/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-PJF?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-PJF/pulls) |
| [**RecSysDatasets**](https://github.com/RUCAIBox/RecSysDatasets) | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecSysDatasets?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecSysDatasets/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecSysDatasets?style=social&logo=github)](https://github.com/RUCAIBox/RecSysDatasets/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecSysDatasets?style=social&logo=git)](https://github.com/RUCAIBox/RecSysDatasets/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecSysDatasets?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecSysDatasets/pulls) |


## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/RecBole/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

We thank the insightful suggestions from [@tszumowski](https://github.com/tszumowski), [@rowedenny](https://github.com/rowedenny), [@deklanw](https://github.com/deklanw) et.al.

We thank the nice contributions through PRs from [@rowedenny](https://github.com/rowedenny)，[@deklanw](https://github.com/deklanw) et.al.


## Cite
If you find RecBole useful for your research or development, please cite the following papers: [RecBole[1.0]](https://arxiv.org/abs/2011.01731), [RecBole[2.0]](https://dl.acm.org/doi/abs/10.1145/3459637.3482016) and [RecBole[1.1.1]](https://arxiv.org/abs/2211.15148).

```bibtex
@inproceedings{recbole[1.0],
  author    = {Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji{-}Rong Wen},
  title     = {RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  booktitle = {{CIKM}},
  pages     = {4653--4664},
  publisher = {{ACM}},
  year      = {2021}
}
@inproceedings{recbole[2.0],
  title={RecBole 2.0: Towards a More Up-to-Date Recommendation Library},
  author={Zhao, Wayne Xin and Hou, Yupeng and Pan, Xingyu and Yang, Chen and Zhang, Zeyu and Lin, Zihan and Zhang, Jingsen and Bian, Shuqing and Tang, Jiakai and Sun, Wenqi and others},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={4722--4726},
  year={2022}
}
@misc{recbole[1.1.1],
  author = {Xu, Lanling and Tian, Zhen and Zhang, Gaowei and Wang, Lei and Zhang, Junjie and Zheng, Bowen and Li, Yifan and Hou, Yupeng and Pan, Xingyu and Chen, Yushuo and Zhao, Wayne Xin and Chen, Xu and Wen, Ji-Rong},
  title = {Recent Advances in RecBole: Extensions with more Practical Considerations},
  journal   = {arXiv preprint arXiv:2211.15148},
  year = {2022}
}
```


## The Team

RecBole is developed by [RUC, BUPT, ECNU](https://www.recbole.io/about.html), and maintained by RUC.

Here is the list of our lead developers in each development phase. They are the souls of RecBole and have made outstanding contributions.

|         Time          |        Version         |                Lead Developers                 |                Paper            |
| :-------------------: | :--------------------: | :--------------------------------------------: | ---------------------------------------------- |
| June 2020<br> ~<br> Nov. 2020 |        v0.1.1         |  Shanlei Mu ([@ShanleiMu](https://github.com/ShanleiMu)), Yupeng Hou ([@hyp1231](https://github.com/hyp1231)),<br> Zihan Lin ([@linzihan-backforward](https://github.com/linzihan-backforward)), Kaiyuan Li ([@tsotfsk](https://github.com/tsotfsk))| [PDF](https://dl.acm.org/doi/abs/10.1145/3459637.3482016) |
|    Nov. 2020<br> ~ <br> Oct. 2022    | v0.1.2 ~ v1.0.1 |      Yushuo Chen ([@chenyushuo](https://github.com/chenyushuo)), Xingyu Pan ([@2017pxy](https://github.com/2017pxy))    | [PDF](https://dl.acm.org/doi/abs/10.1145/3459637.3482016)  |
| Oct. 2022<br/> ~ <br/> now | v1.1.0 ~ v1.1.1 | Lanling Xu ([@Sherry-XLL](https://github.com/Sherry-XLL)), Zhen Tian ([@chenyuwuxin](https://github.com/chenyuwuxin)), Gaowei Zhang ([@Wicknight](https://github.com/Wicknight)), Lei Wang ([@Paitesanshi](https://github.com/Paitesanshi)), Junjie Zhang ([@leoleojie](https://github.com/leoleojie)) | [PDF](https://arxiv.org/abs/2211.15148) |


## License
RecBole uses [MIT License](./LICENSE). All data and code in this project can only be used for academic purposes.