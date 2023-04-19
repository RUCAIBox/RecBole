FM2
===========

Introduction
---------------------

`[paper] <https://arxiv.org/pdf/2102.12994.pdf>`_

**Title:** FM^2: Field-matrixed Factorization Machines for Recommender Systems

**Authors:** Yang Sun, Junwei Pan, Alex Zhang, Aaron Flores

**Abstract:** Click-through rate (CTR) prediction plays a critical role in recom-
mender systems and online advertising. The data used in these
applications are multi-field categorical data, where each feature
belongs to one field. Field information is proved to be important
and there are several works considering fields in their models. In
this paper, we proposed a novel approach to model the field in-
formation effectively and efficiently. The proposed approach is
a direct improvement of FwFM, and is named as Field-matrixed
Factorization Machines (FmFM, or 𝐹 𝑀 2 ). We also proposed a new
explanation of FM and FwFM within the FmFM framework, and
compared it with the FFM. Besides pruning the cross terms, our
model supports field-specific variable dimensions of embedding
vectors, which acts as a soft pruning. We also proposed an efficient
way to minimize the dimension while keeping the model perfor-
mance. The FmFM model can also be optimized further by caching
the intermediate vectors, and it only takes thousands floating-point
operations (FLOPs) to make a prediction. Our experiment results
show that it can out-perform the FFM, which is more complex. The
FmFM model’s performance is also comparable to DNN models
which require much more FLOPs in runtime.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of users, items and entities. Defaults to ``10``.

**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='FM2', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

    learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]

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