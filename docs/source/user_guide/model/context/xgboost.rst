XGBOOST(External algorithm library)
=====================================

Introduction
---------------------

`[XGBoost] <https://xgboost.readthedocs.io/en/latest/>`_

**XGBoost** is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``convert_token_to_onehot (bool)`` : If True, the token type features will be converted into onehot form. Defaults to ``False``.
- ``token_num_threhold (int)`` : The threshold of doing onehot conversion.

- ``xgb_silent (bool, optional)`` : Whether print messages during construction.
- ``xgb_nthread (int, optional)`` : Number of threads to use for loading data when parallelization is applicable. If -1, uses maximum threads available on the system.
- ``xgb_model (file name of stored xgb model or 'Booster' instance)`` :Xgb model to be loaded before training.
- ``xgb_params (dict)`` : Booster params.
- ``xgb_num_boost_round (int)`` : Number of boosting iterations.
- ``xgb_early_stopping_rounds (int)`` : Activates early stopping.
- ``xgb_verbose_eval (bool or int)`` : If verbose_eval is True then the evaluation metric on the validation set is printed at each boosting stage. If verbose_eval is an integer then the evaluation metric on the validation set is printed at every given verbose_eval boosting stage.

Please refer to [XGBoost Python package](https://xgboost.readthedocs.io/en/latest/python/python_api.html) for more details.

**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='xgboost', dataset='ml-100k')

And then:

.. code:: bash

   python run.py
 

If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../user_guide/config_settings`
- :doc:`../../../user_guide/data_intro`
- :doc:`../../../user_guide/train_eval_intro`
- :doc:`../../../user_guide/usage`