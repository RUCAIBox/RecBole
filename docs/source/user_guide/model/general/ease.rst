EASE
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/3308558.3313710>`_

**Title:** Embarrassingly Shallow Autoencoders for Sparse Data

**Authors:** Harald Steck

**Abstract:** Combining simple elements from the literature, we define a linear model that is geared toward sparse data, in particular implicit
feedback data for recommender systems. We show that its training objective has a closed-form solution, and discuss the resulting
conceptual insights. Surprisingly, this simple model achieves better ranking accuracy than various state-of-the-art collaborative
filtering approaches, including deep non-linear models, on most of the publicly available data-sets used in our experiments.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``reg_weight (float)`` : The L2 regularization weight. Defaults to ``250.0``.



**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='EASE', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   reg_weight choice [1.0,10.0,100.0,250.0,500.0,1000.0]

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