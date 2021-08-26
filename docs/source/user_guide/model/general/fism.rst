FISM
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/2487575.2487589>`_

**Title:** FISM: Factored Item Similarity Models for Top-N Recommender Systems

**Authors:** Santosh Kabbur, Xia Ning, George Karypis

**Abstract:** The effectiveness of existing top-N recommendation methods decreases as
the sparsity of the datasets increases. To alleviate this problem, we present an
item-based method for generating top-N recommendations that learns the itemitem
similarity matrix as the product of two low dimensional latent factor matrices.
These matrices are learned using a structural equation modeling approach, wherein the
value being estimated is not used for its own estimation. A comprehensive set of
experiments on multiple datasets at three different sparsity levels indicate that
the proposed methods can handle sparse datasets effectively and outperforms other
state-of-the-art top-N recommendation methods. The experimental results also show
that the relative performance gains compared to competing methods increase as the
data gets sparser.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of users and items. Defaults to ``64``.
- ``split_to (int)`` : This is a parameter used to reduce the GPU memory usage during the evaluation. The larger the value, the less the memory usage and the slower the evaluation speed. Defaults to ``0``.
- ``alpha (float)`` : It is a hyper-parameter controlling the normalization effect of the number of user history interactions when calculating the similarity. Defaults to ``0``.
- ``reg_weights (list)`` : The L2 regularization weights. Defaults to ``[1e-2, 1e-2]``.



**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='FISM', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   reg_weights choice ['[1e-7, 1e-7]','[0, 0]'] 
   alpha choice [0]
   weight_size choice [64]
   beta choice [0.5]

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