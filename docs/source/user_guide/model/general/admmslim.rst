ADMMSLIM
============

Introduction
------------------

`[paper] <https://doi.org/10.1145/3336191.3371774>`_

**Title:** ADMM SLIM: Sparse Recommendations for Many Users

**Authors:** Harald Steck,Maria Dimakopoulou,Nickolai Riabov,Tony Jebara


**Abstract:** The Sparse Linear Method (Slim) is a well-established approach
for top-N recommendations. This article proposes several improvements
that are enabled by the Alternating Directions Method of
Multipliers (ADMM), a well-known optimization method
with many application areas. First, we show that optimizing the
original Slim-objective by ADMM results in an approach where the
training time is independent of the number of users in the training
data, and hence trivially scales to large numbers of users. Second,
the flexibility of ADMM allows us to switch on and off the various
constraints and regularization terms in the original Slim-objective,
in order to empirically assess their contributions to ranking accuracy
on given data. Third, we also propose two extensions to the
original Slim training-objective in order to improve recommendation
accuracy further without increasing the computational cost. In
our experiments on three well-known data-sets, we first compare
to the original Slim-implementation and find that not only ADMM
reduces training time considerably, but also achieves an improvement
in recommendation accuracy due to better optimization. We
then compare to various state-of-the-art approaches and observe
up to 25% improvement in recommendation accuracy in our experiments.
Finally, we evaluate the importance of sparsity and the
non-negativity constraint in the original Slim-objective with subsampling
experiments that simulate scenarios of cold-starting and
large catalog sizes compared to relatively small user base, which
often occur in practice.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``lambda1 (float)`` : L1-norm regularization parameter. Defaults to ``3``.

- ``lambda2 (float)`` : L2-norm regularization parameter. Defaults to ``200``.

- ``alpha (float)`` : The exponents to control the power-law in the regularization terms. Defaults to ``0.5``.

- ``rho (float)`` : The penalty parameter that applies to the squared difference between primal variables. Defaults to ``4000``.

- ``k (int)`` : The number of running iterations. Defaults to ``100``.

- ``positive_only (bool)`` : Whether only preserves all positive values. Defaults to ``True``.

- ``center_columns (bool)`` : Whether to use additional item-bias terms.. Defaults to ``False``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='ADMMSLIM', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

    lambda1 choice [0.1 , 0.5 , 5 , 10]
    lambda2 choice [5 , 50 , 1000 , 5000]
    alpha choice [0.25 , 0.5 , 0.75 , 1]

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