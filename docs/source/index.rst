.. RecBole documentation master file.
.. title:: RecBole v1.0.1
.. image:: asset/logo.png

=========================================================

`HomePage <https://recbole.io/>`_ | `Docs <https://recbole.io/docs/>`_ | `GitHub <https://github.com/RUCAIBox/RecBole>`_ | `Datasets <https://github.com/RUCAIBox/RecDatasets>`_ | `v0.1.2 </docs/v0.1.2/>`_ | `v0.2.0 </docs/v0.2.0/>`_ | `v1.0.0 </docs/v1.0.0/>`_

Introduction
-------------------------
RecBole is a unified, comprehensive and efficient framework developed based on PyTorch.
It aims to help the researchers to reproduce and develop recommendation models.

In the lastest release, our library includes 77 recommendation algorithms `[Model List]`_, covering four major categories:

- General Recommendation
- Sequential Recommendation
- Context-aware Recommendation
- Knowledge-based Recommendation

We design a unified and flexible data file format, and provide the support for 28 benchmark recommendation datasets `[Collected Datasets]`_. A user can apply the provided script to process the original data copy, or simply download the processed datasets by our team.

.. image:: asset/framework.png
    :width: 600
    :align: center

Features:

- General and extensible data structure
    We deign general and extensible data structures to unify the formatting and usage of various recommendation datasets.
- Comprehensive benchmark models and datasets
    We implement 77 commonly used recommendation algorithms, and provide the formatted copies of 28 recommendation datasets.
- Efficient GPU-accelerated execution
    We design many tailored strategies in the GPU environment to enhance the efficiency of our library.
- Extensive and standard evaluation protocols
    We support a series of commonly used evaluation protocols or settings for testing and comparing recommendation algorithms.

.. _[Collected Datasets]:
    /dataset_list.html

.. _[Model List]:
    /model_list.html


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install
   get_started/quick_start

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/config_settings
   user_guide/data_intro
   user_guide/model_intro
   user_guide/train_eval_intro
   user_guide/usage


.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/customize_models
   developer_guide/customize_trainers
   developer_guide/customize_dataloaders
   developer_guide/customize_samplers
   developer_guide/customize_metrics


.. toctree::
   :maxdepth: 1
   :caption: API REFERENCE:

   recbole/recbole.config.configurator
   recbole/recbole.data
   recbole/recbole.evaluator
   recbole/recbole.model
   recbole/recbole.quick_start.quick_start
   recbole/recbole.sampler.sampler
   recbole/recbole.trainer.hyper_tuning
   recbole/recbole.trainer.trainer
   recbole/recbole.utils.case_study
   recbole/recbole.utils.enum_type
   recbole/recbole.utils.logger
   recbole/recbole.utils.utils

The Team
------------------
RecBole is developed and maintained by `RUC, BUPT, ECNU  <https://www.recbole.io/about.html>`_.

Here is the list of our lead developers in each development phase. They are the souls of RecBole and have made outstanding contributions.

======================   ===============   =============================================
Time                     Version 	        Lead Developers
======================   ===============   =============================================
June 2020 ~ Nov. 2020    v0.1.1            `Shanlei Mu <https://github.com/ShanleiMu>`_, `Yupeng Hou <https://github.com/hyp1231>`_, `Zihan Lin <https://github.com/linzihan-backforward>`_, `Kaiyuan Li <https://github.com/tsotfsk>`_
Nov. 2020 ~ Now          v0.1.2 ~ v1.0.0   `Yushuo Chen <https://github.com/chenyushuo>`_, `Xingyu Pan <https://github.com/2017pxy>`_
======================   ===============   =============================================

License
------------
RecBole uses `MIT License <https://github.com/RUCAIBox/RecBole/blob/master/LICENSE>`_.
