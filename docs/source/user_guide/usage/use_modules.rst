Use Modules
================
You can recall different modules in RecBole to satisfy your requirement.

The complete process is as follows:

.. code:: python

    from logging import getLogger
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.general_recommender import BPR
    from recbole.trainer import Trainer
    from recbole.utils import init_seed, init_logger

    if __name__ == '__main__':

        # configurations initialization
        config = Config(model='BPR', dataset='ml-100k')

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

        # model loading and initialization
        model = BPR(config, train_data.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = Trainer(config, model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

        # model evaluation
        test_result = trainer.evaluate(test_data)
        print(test_result)


Configurations Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    config = Config(model='BPR', dataset='ml-100k')

:class:`~recbole.config.configurator.Config` module is used to set parameters and experiment setup. 　
Please refer to :doc:`../config_settings` for more details.


Init Random Seed
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    init_seed(config['seed'], config['reproducibility'])

Initializing the random seed to ensure the reproducibility of the experiments.


Dataset Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    dataset = create_dataset(config)

Filtering the data files according to the parameters indicated in the configuration.


Dataset Splitting
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    train_data, valid_data, test_data = data_preparation(config, dataset)

Splitting the dataset according to the parameters indicated in the configuration.


Model Initialization
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = BPR(config, train_data.dataset).to(config['device'])

Initializing the model according to the model names, and initializing the instance of the model.


Trainer Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python
　　
    trainer = Trainer(config, model)

Initializing the trainer, which is used to model training and evaluation.


Automatic Selection of Model and Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the above example, we manually import the model class :class:`~recbole.model.general_recommender.bpr.BPR` and the trainer class :class:`~recbole.trainer.trainer.Trainer`.
For the implemented model, we support the automatic acquisition of the corresponding model class and
trainer class through the model name.


.. code:: python

    from recbole.utils import get_model, get_trainer

    if __name__ == '__main__':

        ...

        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])

        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        ...


Model Training
^^^^^^^^^^^^^^^^^^^

.. code:: python

    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

Inputting the training and valid data, and beginning the training process.


Model Evaluation
^^^^^^^^^^^^^^^^^^^^^^^
.. code:: python

    test_result = trainer.evaluate(test_data)

Inputting the test data, and evaluating based on the trained model.


Resume Model From Break Point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Our toolkit also supports reloading the parameters from previously trained models.

In this example, we present how to train the model from the former parameters.

.. code:: python

    ...

    if __name__ == '__main__':

        ...

        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        # resume from break point
        checkpoint_file = 'checkpoint.pth'
        trainer.resume_checkpoint(checkpoint_file)

        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

        ...

:attr:`checkpoint_file` is the file used to store the model.


In this example, we present how to test a model based on the previous saved parameters.

.. code:: python

    ...

    if __name__ == '__main__':

        ...

        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        # model evaluation
        checkpoint_file = 'checkpoint.pth'
        test_result = trainer.evaluate(test_data, model_file=checkpoint_file)
        print(test_result)
        ...