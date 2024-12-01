AsymKNN
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/pdf/10.1145/2507157.25071896>`_

**Title:** Efficient Top-N Recommendation for Very Large Scale Binary Rated Datasets

**Authors:** Fabio Aiolli

**Abstract:** We present a simple and scalable algorithm for top-N recommendation able to deal with very large datasets and (binary rated) implicit feedback. We focus on memory-based collaborative filtering
algorithms similar to the well known neighboor based technique for explicit feedback. The major difference, that makes the algorithm particularly scalable, is that it uses positive feedback only
and no explicit computation of the complete (user-by-user or itemby-item) similarity matrix needs to be performed.
The study of the proposed algorithm has been conducted on data from the Million Songs Dataset (MSD) challenge whose task was to suggest a set of songs (out of more than 380k available songs) to more than 100k users given half of the user listening history and
complete listening history of other 1 million people.
In particular, we investigate on the entire recommendation pipeline, starting from the definition of suitable similarity and scoring functions and suggestions on how to aggregate multiple ranking strategies to define the overall recommendation. The technique we are
proposing extends and improves the one that already won the MSD challenge last year.

In this article, we introduce a versatile class of recommendation algorithms that calculate either user-to-user or item-to-item similarities as the foundation for generating recommendations. This approach enables the flexibility to switch between UserKNN and ItemKNN models depending on the desired application.

A distinguishing feature of this class of algorithms, exemplified by AsymKNN, is its use of asymmetric cosine similarity, which generalizes the traditional cosine similarity. Specifically, when the asymmetry parameter
``alpha = 0.5``, the method reduces to the standard cosine similarity, while other values of ``alpha`` allow for tailored emphasis on specific aspects of the interaction data. Furthermore, setting the parameter
``beta = 1.0`` ensures a traditional UserKNN or ItemKNN, as the final scores are only divided by a fixed positive constant, preserving the same order of recommendations.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``k (int)`` : The neighborhood size. Defaults to ``100``.

- ``alpha (float)`` : Weight parameter for asymmetric cosine similarity. Defaults to ``0.5``.

- ``beta (float)`` : Parameter for controlling the balance between factors in the final score normalization. Defaults to ``1.0``.

- ``q (int)`` : The 'locality of scoring function' parameter. Defaults to ``1``.

**Additional Parameters:**

- ``knn_method (str)`` : Calculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.. Defaults to ``item``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='AsymKNN', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   k choice [10,50,100,200,250,300,400,500,1000,1500,2000,2500]
   alpha choice [0.0,0.2,0.5,0.8,1.0]
   beta choice [0.0,0.2,0.5,0.8,1.0]
   q choice [1,2,3,4,5,6]

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