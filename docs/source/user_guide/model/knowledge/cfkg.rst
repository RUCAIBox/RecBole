CFKG
===========

Introduction
---------------------

`[paper] <https://www.mdpi.com/1999-4893/11/9/137>`_

**Title:** Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation

**Authors:** Qingyao Ai, Vahid Azizi, Xu Chen and Yongfeng Zhang

**Abstract:** Providing model-generated explanations in recommender systems is important to user
experience. State-of-the-art recommendation algorithms—especially the collaborative filtering
(CF)-based approaches with shallow or deep models—usually work with various unstructured
information sources for recommendation, such as textual reviews, visual images, and various implicit or
explicit feedbacks. Though structured knowledge bases were considered in content-based approaches,
they have been largely ignored recently due to the availability of vast amounts of data and the learning
power of many complex models. However, structured knowledge bases exhibit unique advantages
in personalized recommendation systems. When the explicit knowledge about users and items is
considered for recommendation, the system could provide highly customized recommendations based
on users’ historical behaviors and the knowledge is helpful for providing informed explanations
regarding the recommended items. A great challenge for using knowledge bases for recommendation is
how to integrate large-scale structured and unstructured data, while taking advantage of collaborative
filtering for highly accurate performance. Recent achievements in knowledge-base embedding (KBE)
sheds light on this problem, which makes it possible to learn user and item representations while
preserving the structure of their relationship with external knowledge for explanation. In this work,
we propose to explain knowledge-base embeddings for explainable recommendation. Specifically,
we propose a knowledge-base representation learning framework to embed heterogeneous entities for
recommendation, and based on the embedded knowledge base, a soft matching algorithm is proposed
to generate personalized explanations for the recommended items. Experimental results on real-world
e-commerce datasets verified the superior recommendation performance and the explainability power
of our approach compared with state-of-the-art baselines.


Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of users, items, entities and relations. Defaults to ``64``.
- ``loss_function (str)`` : The optimization loss function. Defaults to ``'inner_product'``. Range in ``['inner_product', 'transe']``.
- ``margin (float)`` : The margin in margin loss, only be used when ``loss_function`` is set to ``'transe'``. Defaults to ``1.0``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='CFKG', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   loss_function choice ['inner_product', 'transe']
   margin choice [0.5,1.0,2.0]

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