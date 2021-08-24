SLIMElastic
=============

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1109/ICDM.2011.134>`_

**Title:** SLIM: Sparse Linear Methods for Top-N Recommender Systems

**Authors:** Xia Ning and George Karypis

**Abstract:** This paper focuses on developing effective and efficient algorithms for top-N recommender systems. A novel
Sparse LInear Method (SLIM) is proposed, which generates top-N recommendations by aggregating from user purchase/rating
profiles. A sparse aggregation coefficient matrix W is learned from SLIM by solving an L1-norm and L2-norm regularized
optimization problem. W is demonstrated to produce high quality recommendations and its sparsity allows SLIM to generate
recommendations very fast. A comprehensive set of experiments is conducted by comparing the SLIM method and other
state-of-the-art top-N recommendation methods. The experiments show that SLIM achieves significant improvements both
in run time performance and recommendation quality over the best existing methods.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``alpha (float)`` : Control the weights of L1-norm and L2-norm. Defaults to ``0.2``.
- ``l1_ratio (float)`` : Control the weights of L1-norm and L2-norm. The weight of L1-norm used in ElasticNet is equal to ``alpha * l1_ratio``. The weight of L2-norm used in ElasticNet is ``1/2 * alpha * (1 - l1_ratio)``. Defaults to ``0.02``.
- ``positive_only (bool)`` : Whether to add the non-negativity constraint. Defaults to ``True``.
- ``hide_item (bool)`` : Whether to ignore the influence of the item itself. Defaults to ``True``.



**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='SLIMElastic', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   alpha choice [0.2,0.5,0.8,1.0]
   l1_ratio choice [0.5,0.1,0.05,0.02,0.01]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of RecBole (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	python run_hyper.py --model=[model_name] --dataset=[dataset_name] --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`../../../user_guide/usage/parameter_tuning`.


If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../user_guide/config_settings`
- :doc:`../../../user_guide/data_intro`
- :doc:`../../../user_guide/train_eval_intro`
- :doc:`../../../user_guide/usage`