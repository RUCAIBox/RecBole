Introduction
==============

RecBole is a unified, comprehensive and efficient framework developed based on PyTorch.
It aims to help the researchers to reproduce and develop recommendation models.

In the first release, our library includes 73 recommendation algorithms `[Model List]`_, covering four major categories:

- General Recommendation
- Sequential Recommendation
- Context-aware Recommendation
- Knowledge-based Recommendation

We design a unified and flexible data file format, and provide the support for 28 benchmark recommendation datasets `[Collected Datasets]`_. A user can apply the provided script to process the original data copy, or simply download the processed datasets by our team.

Features:

- General and extensible data structure
    We deign general and extensible data structures to unify the formatting and usage of various recommendation datasets.
- Comprehensive benchmark models and datasets
    We implement 73 commonly used recommendation algorithms, and provide the formatted copies of 28 recommendation datasets.
- Efficient GPU-accelerated execution
    We design many tailored strategies in the GPU environment to enhance the efficiency of our library.
- Extensive and standard evaluation protocols
    We support a series of commonly used evaluation protocols or settings for testing and comparing recommendation algorithms.

.. _[Collected Datasets]:
    /dataset_list.html

.. _[Model List]:
    /model_list.html
