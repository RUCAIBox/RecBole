GRU4RecKG
===========

Introduction
---------------------

It is an extension of GRU4Rec, which concatenates items and its corresponding knowledge graph embedding as the input.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of items and the KG feature. Defaults to ``64``.
- ``hidden_size (int)`` : The number of features in the hidden state. Defaults to ``128``.
- ``num_layers (int)`` : The number of layers in GRU. Defaults to ``1``.
- ``dropout_prob (float)`` : The dropout rate. Defaults to ``0.1``.
- ``freeze_kg (bool)`` : Whether to freeze the pre-trained knowledge embedding feature. Defaults to ``True``.
- ``loss_type (str)`` : The type of loss function. If it is set to ``'CE'``, the training task is regarded as a multi-classification task and the target item is the ground truth. In this way, negative sampling is not needed. If it is set to ``'BPR'``, the training task will be optimized in the pair-wise way, which maximizes the difference between the positive item and the negative one. In this way, negative sampling is necessary, such as setting ``--train_neg_sample_args="{'distribution': 'uniform', 'sample_num': 1}"``. Defaults to ``'CE'``. Range in ``['BPR', 'CE']``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   parameter_dict = {
      'train_neg_sample_args': None,
   }
   run_recbole(model='GRU4RecKG', dataset='ml-100k', config_dict=parameter_dict)

And then:

.. code:: bash

   python run.py

**Notes:**

- If you want to run GRU4RecKG, please prepare pretrained knowledge graph embedding and add the following settings to config files:

   .. code:: yaml

        load_col:
            inter: [user_id, item_id]
            kg: [head_id, relation_id, tail_id]
            link: [item_id, entity_id]
            ent_feature: [ent_id, ent_vec]
        alias_of_entity_id: [ent_id]
        preload_weight:
            ent_id: ent_vec
        additional_feat_suffix: [ent_feature]

  where the pretrained knowledge graph embedding should be stored in file named [dataset_name].ent_feature. If you want to
  add additional feature embedding, please refer to this example.

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   dropout_prob choice [0.0,0.1,0.2,0.3,0.4,0.5]
   num_layers choice [1,2,3]
   hidden_size choice [128]
   freeze_kg choice [True, False]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of RecBole (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	python run_hyper.py --model=[model_name] --dataset=[dataset_name] --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`../../../user_guide/usage/parameter_tuning`.


If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../user_guide/config_settings`
- :doc:`../../../user_guide/data_intro`
- :doc:`../../../user_guide/train_eval_intro`
- :doc:`../../../user_guide/usage`