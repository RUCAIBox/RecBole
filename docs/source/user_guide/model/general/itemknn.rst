ItemKNN
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/963770.963776>`_

**Title:** Item-based top-N recommendation algorithms

**Authors:** Mukund Deshpande and George Karypis

**Abstract:** The explosive growth of the world-wide-web and the emergence of e-commerce has led to the development of recommender systems—a personalized information filtering technology used to identify
a set of items that will be of interest to a certain user. User-based collaborative filtering is the most
successful technology for building recommender systems to date and is extensively used in many
commercial recommender systems. Unfortunately, the computational complexity of these methods
grows linearly with the number of customers, which in typical commercial applications can be several millions. To address these scalability concerns model-based recommendation techniques have
been developed. These techniques analyze the user–item matrix to discover relations between the
different items and use these relations to compute the list of recommendations.

In this article, we present one such class of model-based recommendation algorithms that first
determines the similarities between the various items and then uses them to identify the set of
items to be recommended. The key steps in this class of algorithms are (i) the method used to
compute the similarity between the items, and (ii) the method used to combine these similarities
in order to compute the similarity between a basket of items and a candidate recommender item.
Our experimental evaluation on eight real datasets shows that these item-based algorithms are
up to two orders of magnitude faster than the traditional user-neighborhood based recommender
systems and provide recommendations with comparable or better quality.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``k (int)`` : The neighborhood size. Defaults to ``100``.

- ``shrink (float)`` : A normalization hyper parameter in calculate cosine distance. Defaults to ``0.0``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='ItemKNN', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   k choice [10,50,100,200,250,300,400,500,1000,1500,2000,2500] 
   shrink choice [0.0,1.0]

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