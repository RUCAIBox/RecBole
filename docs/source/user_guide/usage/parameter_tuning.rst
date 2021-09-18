Parameter Tuning
=====================
RecBole is featured in the capability of automatic parameter
(or hyper-parameter) tuning. One can readily optimize
a given model according to the provided hyper-parameter spaces.

The general steps are given as follows:

To begin with, the user has to claim a
:class:`~recbole.trainer.hyper_tuning.HyperTuning`
instance in the running python file (e.g., `run.py`):

.. code:: python

    from recbole.trainer import HyperTuning
    from recbole.quick_start import objective_function

    hp = HyperTuning(objective_function=objective_function, algo='exhaustive',
                    params_file='model.hyper', fixed_config_file_list=['example.yaml'])

:attr:`objective_function` is the optimization objective,
the input of :attr:`objective_function` is the parameter,
and the output is the optimal result of these parameters.
The users can design this :attr:`objective_function` according to their own requirements.
The user can also use an encapsulated :attr:`objective_function`, that is:

.. code:: python

    def objective_function(config_dict=None, config_file_list=None):

        config = Config(config_dict=config_dict, config_file_list=config_file_list)
        init_seed(config['seed'])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = get_model(config['model'])(config, train_data).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
        test_result = trainer.evaluate(test_data)

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }

:attr:`algo` is the optimization algorithm. RecBole realizes this module based
on hyperopt_. In addition, we also support grid search tunning method.

.. code:: python

    from hyperopt import tpe

    # hyperopt 自带的优化算法
    hp1 = HyperTuning(algo=tpe.suggest)

    # Grid Search
    hp2 = HyperTuning(algo='exhaustive')

:attr:`params_file` is the ranges of the parameters, which is exampled as
(e.g., `model.hyper`):

.. code:: none

    learning_rate loguniform -8,0
    embedding_size choice [64,96,128]
    mlp_hidden_size choice ['[64,64,64]','[128,128]']

Each line represents a parameter and the corresponding search range.
There are three components: parameter name, range type, range.

:class:`~recbole.trainer.hyper_tuning.HyperTuning` supports four range types,
the details are as follows:

+----------------+---------------------------------+------------------------------------------------------------------+
| range type　   | 　　 range　　　　　　　　　　  | 　　 discription                                                 |
+================+=================================+==================================================================+
| choice         | options(list)                   | search in options                                                |
+----------------+---------------------------------+------------------------------------------------------------------+
| uniform        | low(int),high(int)              | search in uniform distribution: (low,high)                       |
+----------------+---------------------------------+------------------------------------------------------------------+
| loguniform     | low(int),high(int)              | search in uniform distribution: exp(uniform(low,high))           |
+----------------+---------------------------------+------------------------------------------------------------------+
| quniform       | low(int),high(int),q(int)       | search in uniform distribution: round(uniform(low,high)/q)*q     |
+----------------+---------------------------------+------------------------------------------------------------------+

It should be noted that if the parameters are list and the range type is choice,
then the inner list should be quoted, e.g., :attr:`mlp_hidden_size` in `model.hyper`.

.. _hyperopt: https://github.com/hyperopt/hyperopt

:attr:`fixed_config_file_list` is the fixed parameters, e.g., dataset related parameters and evaluation parameters.
These parameters should be aligned with the format in :attr:`config_file_list`. See details as :doc:`../config_settings`.

Calling method of HyperTuning like:

.. code:: python

    from recbole.trainer import HyperTuning
    from recbole.quick_start import objective_function

    hp = HyperTuning(objective_function=objective_function, algo='exhaustive',
                    params_file='model.hyper', fixed_config_file_list=['example.yaml'])

    # run
    hp.run()
    # export result to the file
    hp.export_result(output_file='hyper_example.result')
    # print best parameters
    print('best params: ', hp.best_params)
    # print best result
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

Run like:

.. code:: bash

    python run.py --dataset=[dataset_name] --model=[model_name]

:attr:`dataset_name` is the dataset name, :attr:`model_name` is the model name, which can be controlled by the command line or the yaml configuration files.

For example:

.. code:: yaml

    dataset: ml-100k
    model: BPR

A simple example is to search the :attr:`learning_rate` and :attr:`embedding_size` in BPR, that is,

.. code:: bash

    running_parameters:
    {'embedding_size': 128, 'learning_rate': 0.005}
    current best valid score: 0.3795
    current best valid result:
    {'recall@10': 0.2008, 'mrr@10': 0.3795, 'ndcg@10': 0.2151, 'hit@10': 0.7306, 'precision@10': 0.1466}
    current test result:
    {'recall@10': 0.2186, 'mrr@10': 0.4388, 'ndcg@10': 0.2591, 'hit@10': 0.7381, 'precision@10': 0.1784}

    ...

    best params:  {'embedding_size': 64, 'learning_rate': 0.001}
    best result: {
        'best_valid_result': {'recall@10': 0.2169, 'mrr@10': 0.4005, 'ndcg@10': 0.235, 'hit@10': 0.7582, 'precision@10': 0.1598}
        'test_result': {'recall@10': 0.2368, 'mrr@10': 0.4519, 'ndcg@10': 0.2768, 'hit@10': 0.7614, 'precision@10': 0.1901}
    }
