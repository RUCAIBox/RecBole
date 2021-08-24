Config Introduction
===================
RecBole is able to config different parameters for controlling the experiment
setup (e.g., data processing, data splitting, training and evaluation).
The users can select the settings according to their own requirements.

Config settings
-----------------------------
We split all the config settings into five parts: environment settings, data settings, model settings, training settings and evaluation settings.
The introduction of different parameter configurations are presented as follows (for model settings, please read the specific model page in :doc:`model_intro`):

.. toctree::
   :maxdepth: 1

   config/environment_settings
   config/data_settings
   config/training_settings
   config/evaluation_settings

How to set config?
-----------------------------
RecBole supports three types of parameter configurations: Config files, Parameter Dicts and Command Line. 
The parameters are assigned via the Configuration module.

For more details about setting config, please read 

.. toctree::
   :maxdepth: 1

   config/parameters_configuration


