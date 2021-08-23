Config Settings
===================
RecBole is able to config different parameters for controlling the experiment
setup (e.g., data processing, data splitting, training and evaluation).
The users can select the settings according to their own requirements.

The introduction of different parameter configurations are presented as follows:

Parameters Introduction
-----------------------------
The parameters in RecBole can be divided into three categories:
Basic Parameters, Dataset Parameters and Model Parameters.

Basic Parameters
^^^^^^^^^^^^^^^^^^^^^^
Basic parameters are used to build the general environment including the settings for
model training and evaluation.

**Environment Setting**

- ``gpu_id (int or str)`` : The id of GPU device. Defaults to ``0``.
- ``use_gpu (bool)`` : Whether or not to use GPU. If True, using GPU, else using CPU.
  Defaults to ``True``.
- ``seed (int)`` : Random seed. Defaults to ``2020``.
- ``state (str)`` : Logging level. Defaults to ``'INFO'``.
  Range in ``['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']``.
- ``reproducibility (bool)`` : If True, the tool will use deterministic
  convolution algorithms, which makes the result reproducible. If False,
  the tool will benchmark multiple convolution algorithms and select the fastest one,
  which makes the result not reproducible but can speed up model training in
  some case. Defaults to ``True``.
- ``data_path (str)`` : The path of input dataset. Defaults to ``'dataset/'``.
- ``checkpoint_dir (str)`` : The path to save checkpoint file.
  Defaults to ``'saved/'``.
- ``show_progress (bool)`` : Show the progress of training epoch and evaluate epoch.
  Defaults to ``True``.
- ``save_dataset (bool)``: Whether or not save filtered dataset.
  If True, save filtered dataset, otherwise it will not be saved.
  Defaults to ``False``.
- ``save_dataloaders (bool)``: Whether or not save split dataloaders.
  If True, save split dataloaders, otherwise they will not be saved.
  Defaults to ``False``.

**Training Setting**

- ``epochs (int)`` : The number of training epochs. Defaults to ``300``.
- ``train_batch_size (int)`` : The training batch size. Defaults to ``2048``.
- ``learner (str)`` : The name of used optimizer. Defaults to ``'adam'``.
  Range in ``['adam', 'sgd', 'adagrad', 'rmsprop', 'sparse_adam']``.
- ``learning_rate (float)`` : Learning rate. Defaults to ``0.001``.
- ``training_neg_sample_num (int)`` : The number of negative samples during
  training. If it is set to 0, the negative sampling operation will not be
  performed. Defaults to ``1``.
- ``training_neg_sample_distribution(str)`` : Distribution of the negative items
  in training phase. Default to ``uniform``. Range in ``['uniform', 'popularity']``.
- ``eval_step (int)`` : The number of training epochs before a evaluation
  on the valid dataset. If it is less than 1, the model will not be
  evaluated on the valid dataset. Defaults to ``1``.
- ``stopping_step (int)`` : The threshold for validation-based early stopping.
  Defaults to ``10``.
- ``clip_grad_norm (dict)`` : The args of `clip_grad_norm_ <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`_
  which will clips gradient norm of model. Defaults to ``None``.
- ``loss_decimal_place(int)``: The decimal place of training loss. Defaults to ``4``.
- ``weight_decay (float)`` : Weight decay (L2 penalty), used for `optimizer <https://pytorch.org/docs/stable/optim.html?highlight=weight_decay>`_. Default to ``0.0``.



**Evaluation Setting**

- ``eval_setting (str)``: The evaluation settings. Defaults to ``'RO_RS,full'``.
  The parameter has two parts. The first part control the splitting methods,
  the range is ``['RO_RS','TO_LS','RO_LS','TO_RS']``. The second part(optional)
  control the ranking mechanism, the range is ``['full','uni100','uni1000','pop100','pop1000']``.
- ``group_by_user (bool)``: Whether or not to group the users.
  It must be ``True`` when ``eval_setting`` is in ``['RO_LS', 'TO_LS']``.
  Defaults to ``True``.
- ``split_ratio (list)``: The split ratio between train data, valid data and
  test data. It only take effects when the first part of ``eval_setting``
  is in ``['RO_RS', 'TO_RS']``. Defaults to ``[0.8, 0.1, 0.1]``.
- ``leave_one_num (int)``: It only take effects when the first part of
  ``eval_setting`` is in ``['RO_LS', 'TO_LS']``. Defaults to ``2``.

- ``metrics (list or str)``: Evaluation metrics. Defaults to
  ``['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']``. Range in
  ``['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'AUC', 'GAUC',
  'MAE', 'RMSE', 'LogLoss']``.
- ``topk (list or int or None)``: The value of k for topk evaluation metrics.
  Defaults to ``10``.
- ``valid_metric (str)``: The evaluation metrics for early stopping. 
  It must be one of used ``metrics``. Defaults to ``'MRR@10'``.
- ``eval_batch_size (int)``: The evaluation batch size. Defaults to ``4096``.
- ``metric_decimal_place(int)``: The decimal place of metric score. Defaults to ``4``.

Pleaser refer to :doc:`evaluation_support` for more details about the parameters
in Evaluation Setting.

Dataset Parameters
^^^^^^^^^^^^^^^^^^^^^^^
Dataset Parameters are used to describe the dataset information and control
the dataset loading and filtering.

Please refer to :doc:`data/data_args` for more details.

Model Parameters
^^^^^^^^^^^^^^^^^^^^^
Model Parameters are used to describe the model structures.

Please refer to :doc:`model_intro` for more details.


Parameters Configuration
------------------------------
RecBole supports three types of parameter configurations: Config files,
Parameter Dicts and Command Line. The parameters are assigned via the
Configuration module.

Config Files
^^^^^^^^^^^^^^^^
Config Files should be organized in the format of yaml.
The users should write their parameters according to the rules aligned with
yaml, and the final config files are processed by the configuration module
to complete the parameter settings.

To begin with, we write the parameters into the yaml files (e.g. `example.yaml`).

.. code:: yaml

    gpu_id: 1
    training_batch_size: 1024

Then, the yaml files are conveyed to the configuration module to finish the
parameter settings.

.. code:: python

    from recbole.config import Config

    config = Config(model='BPR', dataset='ml-100k', config_file_list=['example.yaml'])
    print('gpu_id: ', config['gpu_id'])
    print('training_batch_size: ', config['training_batch_size'])


output:

.. code:: bash

    gpu_id: 1
    training_batch_size: 1024

The parameter ``config_file_list`` supports multiple yaml files.

For more details on yaml, please refer to YAML_.

.. _YAML: https://yaml.org/

When using our toolkit, the parameters belonging to **Dataset parameters** and
Evaluation Settings of **Basic Parameters** are recommended to be written into
the config files, which may be convenient for reusing the configurations.

Parameter Dicts
^^^^^^^^^^^^^^^^^^
Parameter Dict is realized by the dict data structure in python, where the key
is the parameter name, and the value is the parameter value. The users can write their
parameters into a dict, and input it into the configuration module.

An example is as follows:

.. code:: python

    from recbole.config import Config

    parameter_dict = {
        'gpu_id': 2,
        'training_batch_size': 512
    }
    config = Config(model='BPR', dataset='ml-100k', config_dict=parameter_dict)
    print('gpu_id: ', config['gpu_id'])
    print('training_batch_size: ', config['training_batch_size'])

output:

.. code:: bash

    gpu_id: 2
    training_batch_size: 512


Command Line
^^^^^^^^^^^^^^^^^^^^^^^^
We can also assign parameters based on the command line.
The parameters in the command line can be read from the configuration module.
The format is: `-–parameter_name=[parameter_value]`.

Write the following code to the python file (e.g. `run.py`):

.. code:: python

    from recbole.config import Config

    config = Config(model='BPR', dataset='ml-100k')
    print('gpu_id: ', config['gpu_id'])
    print('training_batch_size: ', config['training_batch_size'])

Running:

.. code:: bash

    python run.py --gpu_id=3 --training_batch_size=256

output:

.. code:: bash

    gpu_id: 3
    training_batch_size: 256


Priority
^^^^^^^^^^^^^^^^^
RecBole supports the combination of three types of parameter configurations.

The priority of the configuration methods is: Command Line > Parameter Dicts
> Config Files > Default Settings

A example is as follows:

`example.yaml`:

.. code:: yaml

    gpu_id: 1
    training_batch_size: 1024

`run.py`:

.. code:: python

    from recbole.config import Config

    parameter_dict = {
        'gpu_id': 2,
        'training_batch_size': 512
    }
    config = Config(model='BPR', dataset='ml-100k', config_file_list=['example.yaml'], config_dict=parameter_dict)
    print('gpu_id: ', config['gpu_id'])
    print('training_batch_size: ', config['training_batch_size'])

Running:

.. code:: bash

    python run.py --gpu_id=3 --training_batch_size=256

output:

.. code:: bash

    gpu_id: 3
    training_batch_size: 256
