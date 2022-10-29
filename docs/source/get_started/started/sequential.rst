Quick Start: Sequential Recommendation
========================================
For sequential recommendation, we choose **GRU4Rec** model to show you how
to train and test it on the **ml-100k** dataset from both **API** and
**source code**.

Quick-start From API
---------------------

.. _header-n5:

1. Prepare your data:
>>>>>>>>>>>>>>>>>>>>>>>>>

Before running a model, firstly you need to prepare and load data. To
help users quickly get start, RecBole has a build-in dataset **ml-100k**
and you can directly use it. However, if you want to use other datasets,
you can read `Running New
Dataset <https://recbole.io/docs/user_guide/usage/running_new_dataset.html>`__
for more information.

Then, you need to set data config for data loading. You can create a
yaml file called test.yaml and write the following settings:

.. code:: yaml

   # dataset config : Sequential Recommendation
   USER_ID_FIELD: user_id
   ITEM_ID_FIELD: item_id
   TIME_FIELD: timestamp
   load_col:
       inter: [user_id, item_id,timestamp]
   ITEM_LIST_LENGTH_FIELD: item_length
   LIST_SUFFIX: _list
   MAX_ITEM_LIST_LENGTH: 50

As you can see, unlike the example of BPR,sequential recommendation
models utilize the historical interaction sequences to predict the next
item, so it needs to specify and load the user, item and time columns of
the dataset. And you should set the maximum length of the sequence.

For sequential dataset, RecBole supports Augmentation processing. For example,  suppose ``u1`` interacts with ``i1,i2,i3,i4,i5`` in time and ``MAX_ITEM_LIST_LENGTH = 3``

.. code:: python

   user_id:token	item_id_list:token_seq	item_id:token

   0	0 1 2 3 4 	5

After augmentation, sequential dataset will generate these cases.

.. code:: python

   user_id:token	item_id_list:token_seq	item_id:token

   0	1 0 0	2

   0	1 2 0	3
     
   0	1 2 3	4

   0	2 3 4	5


For more details of data config, please refer to `Data
settings <https://recbole.io/docs/user_guide/config/data_settings.html>`__.

.. _header-n11:

2. Choose a model:
>>>>>>>>>>>>>>>>>>>>>>>>>

You can choose a model from our `Model
Introduction <https://recbole.io/docs/user_guide/model_intro.html>`__.
Here we choose GRU4Rec model to demonstrate how to train and test the
sequence recommendation model.

Then, you need to set the parameter for GRU4Rec model. You can check the
`GRU4Rec <https://recbole.io/docs/user_guide/model/sequential/gru4rec.html>`__
and add the model settings into the test.yaml, like:

.. code:: yaml

   # model config
   embedding_size: 64
   hidden_size: 128
   num_layers: 1
   dropout_prob: 0.3
   loss_type: 'CE'

If you want to run different models, you can read `Running Different
Models <https://recbole.io/docs/user_guide/usage/running_different_models.html>`__
for more information.

.. _header-n16:

3. Set training and evaluation config:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

In RecBole, we support multiple training and evaluation methods. You can
choose how to train and test model by simply setting the config.

Here we want to train and test the GRU4Rec model in
training-validation-test method (optimize model parameters on the
training set, do parameter selection according to the results on the
validation set, and finally report the results on the test set) and
evaluate the model performance by full ranking with all item candidates,
so we can add the following settings into the test.yaml.

.. code:: yaml

   # Training and evaluation config
   epochs: 500
   train_batch_size: 4096
   eval_batch_size: 4096
   train_neg_sample_args: ~
   eval_args:
       group_by: user
       order: TO
       split: {'LS': 'valid_and_test'}
       mode: full
   metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
   topk: 10
   valid_metric: MRR@10

For more details of training and evaluation config, please refer to
`Training
Settings <https://recbole.io/docs/user_guide/config/training_settings.html>`__
and `Evaluation
Settings <https://recbole.io/docs/user_guide/config/evaluation_settings.html>`__.

.. _header-n21:

4. Run the model and collect the result
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Now you have finished all the preparations, it’s time to run the model!

You can create a new python file (e.g., run.py), and write the following
code:

.. code:: python

   from recbole.quick_start import run_recbole
   run_recbole(model='GRU4Rec', dataset='ml-100k', config_file_list=['test.yaml'])

Then run the following command:

.. code:: python

   python run.py

And you will obtain the output like:

.. code:: 

   16 Jul 21:12    INFO  ml-100k
   The number of users: 944
   Average actions of users: 106.04453870625663
   The number of items: 1683
   Average actions of items: 59.45303210463734
   The number of inters: 100000
   The sparsity of the dataset: 93.70575143257098%
   Remain Fields: ['user_id', 'item_id', 'timestamp']
   16 Jul 21:12    INFO  [Training]: train_batch_size = [4096] negative sampling: [None]
   16 Jul 21:12    INFO  [Evaluation]: eval_batch_size = [4096] eval_args: [{'group_by': 'user', 'order': 'TO', 'split': {'LS': 'valid_and_test'}, 'mode': 'full'}]
   16 Jul 21:12    INFO  GRU4Rec(
     (item_embedding): Embedding(1683, 64, padding_idx=0)
     (emb_dropout): Dropout(p=0.3, inplace=False)
     (gru_layers): GRU(64, 128, bias=False, batch_first=True)
     (dense): Linear(in_features=128, out_features=64, bias=True)
     (loss_fct): CrossEntropyLoss()
   )
   Trainable parameters: 189696
   Train     0: 100%|█████████████████████████| 24/24 [00:01<00:00, 15.97it/s, GPU RAM: 1.46 G/31.75 G]
   16 Jul 21:12    INFO  epoch 0 training [time: 1.50s, train loss: 176.3402]
   Evaluate   : 100%|██████████████████████████| 1/1 [00:00<00:00, 106.42it/s, GPU RAM: 1.46 G/31.75 G]
   16 Jul 21:12    INFO  epoch 0 evaluating [time: 0.02s, valid_score: 0.008100]
   ......
   Train    43: 100%|█████████████████████████| 24/24 [00:01<00:00, 17.43it/s, GPU RAM: 1.46 G/31.75 G]
   16 Jul 21:13    INFO  epoch 43 training [time: 1.38s, train loss: 134.4222]
   Evaluate   : 100%|███████████████████████████| 1/1 [00:00<00:00, 86.71it/s, GPU RAM: 1.46 G/31.75 G]
   16 Jul 21:13    INFO  epoch 43 evaluating [time: 0.02s, valid_score: 0.043600]
   16 Jul 21:13    INFO  valid result: 
   recall@10 : 0.1326    mrr@10 : 0.0436    ndcg@10 : 0.0641    hit@10 : 0.1326    precision@10 : 0.0133
   16 Jul 21:13    INFO  Finished training, best eval result in epoch 32
   16 Jul 21:13    INFO  Loading model structure and parameters from saved/GRU4Rec-Jul-16-2022_21-12-43.pth
   Evaluate   : 100%|██████████████████████████| 1/1 [00:00<00:00, 238.76it/s, GPU RAM: 1.46 G/31.75 G]
   16 Jul 21:13    INFO  best valid : OrderedDict([('recall@10', 0.1442), ('mrr@10', 0.0501), ('ndcg@10', 0.0717), ('hit@10', 0.1442), ('precision@10', 0.0144)])
   16 Jul 21:13    INFO  test result: OrderedDict([('recall@10', 0.1103), ('mrr@10', 0.0337), ('ndcg@10', 0.0513), ('hit@10', 0.1103), ('precision@10', 0.011)])


Finally you will get the model’s performance on the test set and the
model file will be saved under the /saved. Besides, RecBole allows
tracking and visualizing train loss and valid score with TensorBoard,
please read the `Use
Tensorboard <https://recbole.io/docs/user_guide/usage/use_tensorboard.html>`__
for more details.

The above is the whole process of running a model in RecBole, and you
can read other docs for depth usage.

.. _header-n31:

Quick-start From Source
--------------------------

Besides using API, you can also directly run the source code of
`RecBole <https://github.com/RUCAIBox/RecBole>`__. The whole process is
similar to Quick-start From API. You can create a yaml file called
test.yaml and set all the config as follow:

.. code:: yaml

   # dataset config : Sequential Recommendation
   USER_ID_FIELD: user_id
   ITEM_ID_FIELD: item_id
   TIME_FIELD: timestamp
   load_col:
       inter: [user_id, item_id,timestamp]
   ITEM_LIST_LENGTH_FIELD: item_length
   LIST_SUFFIX: _list
   MAX_ITEM_LIST_LENGTH: 50

   # model config
   embedding_size: 64
   hidden_size: 128
   num_layers: 1
   dropout_prob: 0.3
   loss_type: 'CE'

   # Training and evaluation config
   epochs: 500
   train_batch_size: 4096
   eval_batch_size: 4096
   train_neg_sample_args: ~
   eval_args:
       group_by: user
       order: TO
       split: {'LS': 'valid_and_test'}
       mode: full
   metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
   topk: 10
   valid_metric: MRR@10
   metric_decimal_place: 4

Then run the following command:

.. code:: python

   python run_recbole.py --model=GRU4Rec --dataset=ml-100k --config_files=test.yaml

And you will get the output of running the GRU4Rec model on the ml-100k
dataset.

If you want to change the parameters, such as ``embedding_size``, just
set the additional command parameters as you need:

.. code:: python

   python run_recbole.py --model=GRU4Rec --dataset=ml-100k --config_files=test.yaml --embedding_size=100
