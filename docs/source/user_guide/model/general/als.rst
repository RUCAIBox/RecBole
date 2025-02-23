ALS(External algorithm library)
===========

Introduction
---------------------

`[ALS (implicit)] <https://benfred.github.io/implicit/api/models/cpu/als.html>`_

**ALS (AlternatingLeastSquares)** by implicit is a Recommendation Model based on the algorithm proposed by Koren in `Collaborative Filtering for Implicit Feedback Datasets <http://yifanhu.net/PUB/cf.pdf>`_.
It furthermore leverages the finding out of `Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering <https://dl.acm.org/doi/pdf/10.1145/2043932.2043987>`_ for performance optimization.
`Implicit <https://benfred.github.io/implicit/index.html>`_ provides several models for implicit feedback recommendations.

`[paper] <http://yifanhu.net/PUB/cf.pdf>`_

**Title:** Collaborative Filtering for Implicit Feedback Datasets

**Authors:** Hu, Yifan and Koren, Yehuda and Volinsky, Chris

**Abstract:** A common task of recommender systems is to improve
customer experience through personalized recommendations based on prior implicit feedback. These systems passively track different sorts of user behavior, such as purchase history, watching habits and browsing activity, in order to model user preferences. Unlike the much more extensively researched explicit feedback, we do not have any
direct input from the users regarding their preferences. In
particular, we lack substantial evidence on which products
consumer dislike. In this work we identify unique properties of implicit feedback datasets. We propose treating the
data as indication of positive and negative preference associated with vastly varying confidence levels. This leads to a
factor model which is especially tailored for implicit feedback recommenders. We also suggest a scalable optimization procedure, which scales linearly with the data size. The
algorithm is used successfully within a recommender system
for television shows. It compares favorably with well tuned
implementations of other known methods. In addition, we
offer a novel way to give explanations to recommendations
given by this factor model.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The number of latent factors to compute. Defaults to ``64``.
- ``regularization (float)`` : The regularization factor to use. Defaults to ``0.01``.
- ``alpha (float)`` : The weight to give to positive examples. Defaults to ``1.0``.

Please refer to [Implicit Python package](https://benfred.github.io/implicit/index.html) for more details.

**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='ALS', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

    regularization choice [0.01, 0.03, 0.05, 0.1]
    embedding_size choice [32, 64, 96, 128, 256]
    alpha choice [0.5, 0.7, 1.0, 1.3, 1.5]

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