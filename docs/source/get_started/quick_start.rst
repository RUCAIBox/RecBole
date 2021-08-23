Quick Start
===============
Here is a quick-start example for using RecBole. We will show you how to train and test **BPR** model on the **ml-1m** dataset from API
and source code.


Quick-start From API
--------------------------

1. Prepare data:
>>>>>>>>>>>>>>
Before running a model, firstly you need to prepare and load data. In order to characterize most forms of the input data
required by different recommendation tasks, RecBole designs an input data format called :doc:`../user_guide/data/atomic_files` and 
you need to convert your raw data into Atomic Files before data loading. For the convenience of users, we have collected more than
28 commonly used datasets (detailed as `Dataset List </dataset_list.html>`_.) and released their Atomic Files format 
for users to download them freely. 

Then, you need to set data config for data loading. You can create a `yaml` file called `test.yaml` and write the following settings:

.. code:: yaml

    # dataset config
    data_path: # YOUR_DATA_FILE_PATH
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    load_col:
        inter: [user_id, item_id]

For more details of data config, please refer to :doc:`../user_guide/data/data_settings`.

2. Choose a model:
>>>>>>>>>>>>>>
In RecBole, we implement 72 recommendation models covering general recommendation, sequential recommendation,
context-aware recommendation and knowledge-based recommendation. You can choose a model from our :doc:`../user_guide/model_intro`.
Here we choose BPR model to train and test. 

Then, you need to set the parameter for BPR model. You can check the :doc:`../user_guide/model/general/bpr` and add the model settings into the `test.yaml`, like:

.. code:: yaml

    # model config
    embedding_size: 64

3. Set training and evaluation config:
>>>>>>>>>>>>>>
In RecBole, we support multiple training and evaluation methods. You can choose how to train and test model by simply setting the config.
Here we want to train and test the BPR model in training-validation-test method (optimize model parameters on the training set, do parameter selection according to the results on the validation set,
and finally report the results on the test set) and evaluate the model performance by full ranking with all item candidates, 
so we can add the following settings into the `test.yaml`.

.. code:: yaml

    # Training and evaluation config
    epochs: 500
    train_batch_size: 4096
    eval_batch_size: 4096
    neg_sampling:
        uniform: 1
    eval_args:
        group_by: user
        order: RO
        split: {'RS': [0.8,0.1,0.1]}
        mode: full
    metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
    topk: 10 
    valid_metric: MRR@10
    metric_decimal_place: 4

For more details of training and evaluation config, please refer to :doc:`../user_guide/train_eval/training_settings` and :doc:`../user_guide/train_eval/eval_settings`.

4. Run the model and collect the result
>>>>>>>>>>>>>>
Now you have finished all the preparations, it's time to run the model!
you can create a new python file (e.g., `run.py`), and write the following code:

.. code:: python

    from recbole.quick_start import run_recbole

    run_recbole(model='BPR', dataset='ml-1m')


Then run the following command:

.. code:: bash

    python run.py --config_files=test.yaml

And you will obtain the output like:

.. code:: none

    09 Aug 03:44    INFO  ml-1m
    The number of users: 6041
    Average actions of users: 165.5975165562914
    The number of items: 3707
    Average actions of items: 269.88909875876953
    The number of inters: 1000209
    The sparsity of the dataset: 95.53358229599758%
    Remain Fields: ['user_id', 'item_id']
    09 Aug 03:44    INFO  [Training]: train_batch_size = [2048] negative sampling: [{'uniform': 1}]
    09 Aug 03:44    INFO  [Evaluation]: eval_batch_size = [4096] eval_args: [{'split': {'RS': [0.8, 0.1, 0.1]}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'}]
    09 Aug 03:44    INFO  BPR(
    (user_embedding): Embedding(6041, 64)
    (item_embedding): Embedding(3707, 64)
    (loss): BPRLoss()
    )
    Trainable parameters: 623872
    Train     0: 100%|██████████████████████| 394/394 [00:01<00:00, 220.17it/s, GPU RAM: 0.03 G/11.91 G]
    09 Aug 03:44    INFO  epoch 0 training [time: 1.82s, train loss: 232.5692]
    Evaluate   : 100%|████████████████████| 6040/6040 [00:08<00:00, 672.60it/s, GPU RAM: 0.03 G/11.91 G]
    09 Aug 03:44    INFO  epoch 0 evaluating [time: 9.05s, valid_score: 0.225600]
    ......
    09 Aug 03:51    INFO  Finished training, best eval result in epoch 30
    09 Aug 03:51    INFO  Loading model structure and parameters from saved/BPR-Aug-09-2021_03-44-12.pth
    Evaluate   : 100%|████████████████████| 6040/6040 [00:08<00:00, 718.04it/s, GPU RAM: 0.03 G/11.91 G]
    09 Aug 03:51    INFO  best valid : {'recall@10': 0.1466, 'mrr@10': 0.378, 'ndcg@10': 0.2067, 'hit@10': 0.7278, 'precision@10': 0.1625}
    09 Aug 03:51    INFO  test result: {'recall@10': 0.1614, 'mrr@10': 0.4432, 'ndcg@10': 0.2558, 'hit@10': 0.7422, 'precision@10': 0.201}

Finally you will get the model's performance on the test set and the model file will be saved under the `/save`. Besides, 
RecBole allows tracking and visualizing train loss and valid score with TensorBoard, please read the xxx for more details.

The above is the whole process of running a model in RecBole, and you can read other docs for depth usage. 


Quick-start From Source
--------------------------
Besides using API, you can also directly run the source code of `RecBole <https://github.com/RUCAIBox/RecBole>`_. 
The whole process is similar to Quick-start From API. 
You can create a `yaml` file called `test.yaml` and set all the config as follow:

.. code:: yaml

    # dataset config 
    data_path: # YOUR_DATA_FILE_PATH
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
    neg_sampling:
        uniform: 1
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

    python run_recbole.py --model=BPR --dataset=ml-1m --config_files=test.yaml

And you will get the output of running the BPR model on the ml-1m dataset.

If you want to change the parameters, such as ``embedding_size``,
just set the additional command parameters as you need:

.. code:: bash

    python run_recbole.py --model=BPR --dataset=ml-1m --config_files=test.yaml --embedding_size=0.0001 



In-depth Usage
-------------------
For a more in-depth usage about RecBole, take a look at

- :doc:`../user_guide/config_setting`
- :doc:`../user_guide/data_intro`
- :doc:`../user_guide/model_intro`
- :doc:`../user_guide/evaluation_support`
- :doc:`../user_guide/usage`
