LIGHTGBM(External algorithm library)
=====================================

Introduction
---------------------

`[LightGBM] <https://lightgbm.readthedocs.io/en/latest/>`_

**LightGBM** is a gradient boosting framework that uses tree based learning algorithms.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``convert_token_to_onehot (bool)`` : If True, the token type features will be converted into one-hot form. Defaults to ``False``.
- ``token_num_threhold (int)`` : The threshold of one-hot conversion. Defaults to ``10000``.

- ``lgb_silent (bool, optional)`` : Whether to print messages during construction. Defaults to ``False``.
- ``lgb_model (file name of stored lgb model or 'Booster' instance)`` : Lgb model to be loaded before training. Defaults to ``None``.
- ``lgb_params (dict)`` : Booster params.
- ``lgb_learning_rates (list, callable or None)`` : List of learning rates for each boosting round or a customized function that calculates ``learning_rate`` in terms of current number of round (e.g. yields learning rate decay).  Defaults to ``None``.
- ``lgb_num_boost_round (int)`` : Number of boosting iterations. Defaults to ``300``.
- ``lgb_early_stopping_rounds (int or None)`` : Activates early stopping. The model will train until the validation score stops improving. Validation score needs to improve at least every ``early_stopping_rounds`` round(s) to continue training.  Defaults to ``None``.
- ``lgb_verbose_eval (bool or int)`` : Requires at least one validation data. If True, the eval metric on the valid set is printed at each boosting stage. If int, the eval metric on the valid set is printed at every ``verbose_eval`` boosting stage. The last boosting stage or the boosting stage found by using ``early_stopping_rounds`` is also printed. Defaults to ``100``.

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