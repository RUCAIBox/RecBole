Use run_recbole_group
==========================

We have implemented ``run_recbole_group.py``
to support users to evaluate multiple models in a unified configuration and
display the best results on each evaluation metric.
Meanwhile, we support the conversion of evaluation results into latex code,
so that users can use it conveniently when writing papers.


Like :func:`~recbole.quick_start.quick_start.run_recbole`, you can execute the following command to run:

.. code:: bash

    python run_recbole_group.py --[model_list]=[model_list] --[dataset]=[dataset] --[config_files]=[config_files]
    --[valid_latex]=[valid_latex] --[test_latex]=[test_latex]

`--[model_list]=[model_list]` is used to select the scope of the models for evaluating.
Different models are separated by commas, such as 'BPR,LightGCN'.
`--[dataset]=[dataset]` is the way to control dataset by the command line, such as 'ml-100k'.
All models run on this unified dataset.
`--[config_files]=[config_files]` indicates the configuration files.
Please refer to :doc:`../config_settings` for more details about config settings.
`--[valid_latex]=[valid_latex]` is the way to control
the output path of the latex code for the results on valid dataset.
`--[test_latex]=[test_latex]` is the way to control
the output path of the latex code for the results on test dataset.
