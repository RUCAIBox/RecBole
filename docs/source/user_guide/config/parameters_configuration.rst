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
