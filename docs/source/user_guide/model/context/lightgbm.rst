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

**Elaborated example**

Note: a custom lightgbmTrainer should be imported in order for the code to work!

.. code:: python

   from logging import getLogger
   from recbole.config import Config
   from recbole.data import create_dataset, data_preparation
   from recbole.model.exlib_recommender import lightgbm
   from recbole.trainer import lightgbmTrainer
   from recbole.utils import init_seed, init_logger

   if __name__ == '__main__':
       config = Config(model='lightgbm', dataset='X', config_file_list=['X.yaml'])
       dataset = create_dataset(config)

       # init random seed
       init_seed(config['seed'], config['reproducibility'])

       # logger initialization
       init_logger(config)
       logger = getLogger()

       # write config info into log
       logger.info(config)

       # dataset creating and filtering
       dataset = create_dataset(config)
       logger.info(dataset)

       # dataset splitting
       train_data, valid_data, test_data = data_preparation(config, dataset)

       config['lgb_params'] = {
           'boosting': 'gbdt',
           'objective': 'binary',
           'metric': ['auc', 'binary_logloss']
       }

       config['lgb_num_boost_round'] = 100

       # model loading and initialization
       model = lightgbm(config, train_data).to(config['device'])
       logger.info(model)

       # trainer loading and initialization
       trainer = lightgbmTrainer(config, model)

       # model training
       best_valid_score, best_valid_result = trainer.fit(
           train_data, valid_data, show_progress=True)

       # model evaluation
       test_result = trainer.evaluate(test_data)
       print(test_result)

 

If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../user_guide/config_settings`
- :doc:`../../../user_guide/data_intro`
- :doc:`../../../user_guide/evaluation_support`
- :doc:`../../../user_guide/usage`
