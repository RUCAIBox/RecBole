Quick Start: General Recommendation
=====================================
For general recommendation, we choose **BPR** model to show you how to train
and test it on the **ml-100k** dataset from both **API** and **source
code**.

Quick-start From API
--------------------------

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

   # dataset config : General Recommendation
   USER_ID_FIELD: user_id
   ITEM_ID_FIELD: item_id
   load_col:
       inter: [user_id, item_id]

General recommendation models utilize the historical interactions
between users and items to make recommendations, so it needs to specify
and load the user and item columns of the dataset.

2. Choose a model:
>>>>>>>>>>>>>>>>>>>>>>>>>

You can choose a model from our `Model
Introduction <https://recbole.io/docs/user_guide/model_intro.html>`__.
Here we choose BPR model to demonstrate how to train and test the
knowledge-based Recommendation,model.

Then, you need to set the parameter for BPR model. You can check the
`BPR <https://recbole.io/docs/user_guide/model/general/bpr.html>`__ and
add the model settings into the test.yaml, like:

.. code:: yaml

   # model config
   embedding_size: 64

If you want to run different models, you can read `Running Different
Models <https://recbole.io/docs/user_guide/usage/running_different_models.html>`__
for more information.

3. Set training and evaluation config:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

In RecBole, we support multiple training and evaluation methods. You can
choose how to train and test model by simply setting the config.

Here we want to train and test the BPR model in training-validation-test
method (optimize model parameters on the training set, do parameter
selection according to the results on the validation set, and finally
report the results on the test set) and evaluate the model performance
by full ranking with all item candidates, so we can add the following
settings into the test.yaml.

.. code:: yaml

   # Training and evaluation config
   epochs: 500
   train_batch_size: 4096
   eval_batch_size: 4096
   train_neg_sample_args:
       distribution: uniform
       sample_num: 1
       alpha: 1.0
       dynamic: False
       candidate_num: 0
   eval_args:
       group_by: user
       order: RO
       split: {'RS': [0.8,0.1,0.1]}
       mode: full
   metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
   topk: 10
   valid_metric: MRR@10
   metric_decimal_place: 4

For more details of training and evaluation config, please refer to
`Training
Settings <https://recbole.io/docs/user_guide/config/training_settings.html>`__
and `Evaluation
Settings <https://recbole.io/docs/user_guide/config/evaluation_settings.html>`__.

.. _header-n40:

4. Run the model and collect the result
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Now you have finished all the preparations, it’s time to run the model!

You can create a new python file (e.g., run.py), and write the following
code:

.. code:: python

   from recbole.quick_start import run_recbole
   run_recbole(model='BPR', dataset='ml-100k', config_file_list=['test.yaml'])

Then run the following command:

.. code:: python

   python run.py

And you will obtain the output like:

.. code:: 

   24 Aug 01:46    INFO  ml-100k
   The number of users: 944
   Average actions of users: 106.04453870625663
   The number of items: 1683
   Average actions of items: 59.45303210463734
   The number of inters: 100000
   The sparsity of the dataset: 93.70575143257098%
   Remain Fields: ['user_id', 'item_id']
   24 Aug 01:46    INFO  [Training]: train_batch_size = [4096] negative sampling: [{'uniform': 1}]
   24 Aug 01:46    INFO  [Evaluation]: eval_batch_size = [4096] eval_args: [{'split': {'RS': [0.8, 0.1, 0.1]}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'}]
   24 Aug 01:46    INFO  BPR(
   (user_embedding): Embedding(944, 64)
   (item_embedding): Embedding(1683, 64)
   (loss): BPRLoss()
   )
   Trainable parameters: 168128
   Train     0: 100%|████████████████████████| 40/40 [00:00<00:00, 200.47it/s, GPU RAM: 0.01 G/11.91 G]
   24 Aug 01:46    INFO  epoch 0 training [time: 0.21s, train loss: 27.7228]
   Evaluate   : 100%|██████████████████████| 472/472 [00:00<00:00, 518.65it/s, GPU RAM: 0.01 G/11.91 G]
   24 Aug 01:46    INFO  epoch 0 evaluating [time: 0.92s, valid_score: 0.020500]
   ......
   Train    96: 100%|████████████████████████| 40/40 [00:00<00:00, 229.26it/s, GPU RAM: 0.01 G/11.91 G]
   24 Aug 01:47    INFO  epoch 96 training [time: 0.18s, train loss: 3.7170]
   Evaluate   : 100%|██████████████████████| 472/472 [00:00<00:00, 857.00it/s, GPU RAM: 0.01 G/11.91 G]
   24 Aug 01:47    INFO  epoch 96 evaluating [time: 0.56s, valid_score: 0.375200]
   24 Aug 01:47    INFO  valid result:
   recall@10 : 0.2162    mrr@10 : 0.3752    ndcg@10 : 0.2284    hit@10 : 0.7508    precision@10 : 0.1602
   24 Aug 01:47    INFO  Finished training, best eval result in epoch 85
   24 Aug 01:47    INFO  Loading model structure and parameters from saved/BPR-Aug-24-2021_01-46-43.pth
   Evaluate   : 100%|██████████████████████| 472/472 [00:00<00:00, 866.53it/s, GPU RAM: 0.01 G/11.91 G]
   24 Aug 01:47    INFO  best valid : {'recall@10': 0.2195, 'mrr@10': 0.3871, 'ndcg@10': 0.2344, 'hit@10': 0.7582, 'precision@10': 0.1627}
   24 Aug 01:47    INFO  test result: {'recall@10': 0.2523, 'mrr@10': 0.4855, 'ndcg@10': 0.292, 'hit@10': 0.7953, 'precision@10': 0.1962}

Finally you will get the model’s performance on the test set and the
model file will be saved under the /saved. Besides, RecBole allows
tracking and visualizing train loss and valid score with TensorBoard,
please read the `Use
Tensorboard <https://recbole.io/docs/user_guide/usage/use_tensorboard.html>`__
for more details.

The above is the whole process of running a model in RecBole, and you
can read other docs for depth usage.

.. _header-n28:

Quick-start From Source
--------------------------
Besides using API, you can also directly run the source code of `RecBole <https://github.com/RUCAIBox/RecBole>`_. 
The whole process is similar to Quick-start From API. 
You can create a `yaml` file called `test.yaml` and set all the config as follow:

.. code:: yaml

    # dataset config : General Recommendation
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    load_col:
        inter: [user_id, item_id]
    
    # model config
    embedding_size: 64
    
    # Training and evaluation config
    epochs: 500
    train_batch_size: 4096
    eval_batch_size: 4096
    train_neg_sample_args:
        distribution: uniform
        sample_num: 1
        alpha: 1.0
        dynamic: False
        candidate_num: 0
    eval_args:
        group_by: user
        order: RO
        split: {'RS': [0.8,0.1,0.1]}
        mode: full
    metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
    topk: 10 
    valid_metric: MRR@10
    metric_decimal_place: 4

Then run the following command:

.. code:: bash

    python run_recbole.py --model=BPR --dataset=ml-100k --config_files=test.yaml

And you will get the output of running the BPR model on the ml-100k dataset.

If you want to change the parameters, such as ``embedding_size``,
just set the additional command parameters as you need:

.. code:: bash

    python run_recbole.py --model=BPR --dataset=ml-100k --config_files=test.yaml --embedding_size=100 

