LIGHTGBM(External algorithm library)
=====================================

Introduction
---------------------

`[LightGBM] <https://lightgbm.readthedocs.io/en/latest/>`_

**LightGBM** is a gradient boosting framework that uses tree based learning algorithms.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``convert_token_to_onehot (bool)`` : If True, the token type features will be converted into onehot form. Defaults to ``False``.
- ``token_num_threhold (int)`` : The threshold of doing onehot conversion.

- ``lgb_silent (bool, optional)`` : Whether print messages during construction.
- ``lgb_model (file name of stored lgb model or 'Booster' instance)`` :Lgb model to be loaded before training.
- ``lgb_params (dict)`` : Booster params.
- ``lgb_learning_rates (int)`` : List of learning rates for each boosting round or a customized function that calculates learning_rate in terms of current number of round.
- ``lgb_num_boost_round (int)`` : Number of boosting iterations.
- ``lgb_early_stopping_rounds (int)`` : Activates early stopping.
- ``lgb_verbose_eval (bool or int)`` : If verbose_eval is True then the evaluation metric on the validation set is printed at each boosting stage. If verbose_eval is an integer then the evaluation metric on the validation set is printed at every given verbose_eval boosting stage.

Please refer to [LightGBM Python package](https://lightgbm.readthedocs.io/en/latest/Python-API.html) for more details.

**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='lightgbm', dataset='ml-100k')

And then:

.. code:: bash

   python run.py
 

If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../user_guide/config_settings`
- :doc:`../../../user_guide/data_intro`
- :doc:`../../../user_guide/train_eval_intro`
- :doc:`../../../user_guide/usage`