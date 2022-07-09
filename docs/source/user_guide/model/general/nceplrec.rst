NCEPLRec
============

Introduction
------------------

`[paper] <https://dl.acm.org/doi/10.1145/3331184.3331201>`_

**Title:** Noise Contrastive Estimation for One-Class Collaborative Filtering

**Authors:** Ga Wu, Maksims Volkovs, Chee Loong Soon, Scott Sanner, Himanshu Rai

**Abstract:**
Previous highly scalable One-Class Collaborative Filtering (OC-CF)
methods such as Projected Linear Recommendation (PLRec) have
advocated using fast randomized SVD to embed items into a latent
space, followed by linear regression methods to learn personalized
recommendation models per user. However, naive SVD embedding
methods often exhibit a strong popularity bias that prevents them
from accurately embedding less popular items, which is exacer-
bated by the extreme sparsity of implicit feedback matrices in the
OC-CF setting. To address this deficiency, we leverage insights from
Noise Contrastive Estimation (NCE) to derive a closed-form, effi-
ciently computable “depopularized” embedding. We show that NCE
item embeddings combined with a personalized user model from
PLRec produces superior recommendations that adequately account
for popularity bias. Further analysis of the popularity distribution
of recommended items demonstrates that NCE-PLRec uniformly
distributes recommendations over the popularity spectrum while
other methods exhibit distinct biases towards specific popularity
subranges. Empirically, NCE-PLRec produces highly competitive
performance with run-times an order of magnitude faster than
existing state-of-the-art approaches for OC-CF.



Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``rank (int)`` : The latent dimension of latent representations. Defaults to ``450``.
- ``beta (float)`` : The popularity sensitivity. Defaults to ``1.0``.
- ``reg_weight (float)`` : The regularization weight. Defaults to ``1e-02``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='NCEPLRec', dataset='ml-100k')

And then:

.. code:: bash

   python run.py


Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   rank choice [100,200,450]
   beta choice [0.8,1.0,1.3]
   reg_weight choice [1e-04,1e-02,1e2,15000]
   

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

