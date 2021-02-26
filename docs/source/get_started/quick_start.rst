Quick Start
===============
Here is a quick-start example for using RecBole.

Quick-start From Source
--------------------------
With the source code of `RecBole <https://github.com/RUCAIBox/RecBole>`_,
the following script can be used to run a toy example of our library.

.. code:: bash

    python run_recbole.py

This script will run the BPR model on the ml-100k dataset.

Typically, this example takes less than one minute. We will obtain some output like:

.. code:: none

    INFO ml-100k
    The number of users: 944
    Average actions of users: 106.04453870625663
    The number of items: 1683
    Average actions of items: 59.45303210463734
    The number of inters: 100000
    The sparsity of the dataset: 93.70575143257098%

    INFO Evaluation Settings:
    Group by user_id
    Ordering: {'strategy': 'shuffle'}
    Splitting: {'strategy': 'by_ratio', 'ratios': [0.8, 0.1, 0.1]}
    Negative Sampling: {'strategy': 'full', 'distribution': 'uniform'}

    INFO BPRMF(
        (user_embedding): Embedding(944, 64)
        (item_embedding): Embedding(1683, 64)
        (loss): BPRLoss()
    )
    Trainable parameters: 168128

    INFO epoch 0 training [time: 0.27s, train loss: 27.7231]
    INFO epoch 0 evaluating [time: 0.12s, valid_score: 0.021900]
    INFO valid result:
    recall@10: 0.0073  mrr@10: 0.0219  ndcg@10: 0.0093  hit@10: 0.0795  precision@10: 0.0088

    ...

    INFO epoch 63 training [time: 0.19s, train loss: 4.7660]
    INFO epoch 63 evaluating [time: 0.08s, valid_score: 0.394500]
    INFO valid result:
    recall@10: 0.2156  mrr@10: 0.3945  ndcg@10: 0.2332  hit@10: 0.7593  precision@10: 0.1591

    INFO Finished training, best eval result in epoch 52
    INFO Loading model structure and parameters from saved/***.pth
    INFO best valid result:
    recall@10: 0.2169  mrr@10: 0.4005  ndcg@10: 0.235  hit@10: 0.7582  precision@10: 0.1598
    INFO test result:
    recall@10: 0.2368  mrr@10: 0.4519  ndcg@10: 0.2768  hit@10: 0.7614  precision@10: 0.1901

Note that using the quick start pipeline we provide, the original dataset will be divided into training set, validation set and test set by default.
We optimize model parameters on the training set, do parameter selection according to the results on the validation set,
and finally report the results on the test set.

If you want to change the parameters, such as ``learning_rate``, ``embedding_size``,
just set the additional command parameters as you need:

.. code:: bash

    python run_recbole.py --learning_rate=0.0001 --embedding_size=128


If you want to change the models, just run the script by setting additional command parameters:

.. code:: bash

    python run_recbole.py --model=[model_name]

``model_name`` indicates the model to be initialized.
RecBole has implemented four categories of recommendation algorithms
including general recommendation, context-aware recommendation,
sequential recommendation and knowledge-based recommendation.
More details can be found in :doc:`../user_guide/model_intro`.


The datasets can be changed according to :doc:`../user_guide/data_intro`.


Quick-start From API
-------------------------
If RecBole is installed from ``pip`` or ``conda``, you can create a new python file (e.g., `run.py`),
and write the following code:

.. code:: python

    from recbole.quick_start import run_recbole

    run_recbole()


Then run the following command:

.. code:: bash

    python run.py --dataset=ml-100k --model=BPR

This will perform the training and test of the BPR model on the ml-100k dataset.

One can also use similar methods as mentioned above to run different models, parameters or datasets,
the operations are same with `Quick-start From Source`_.


In-depth Usage
-------------------
For a more in-depth usage about RecBole, take a look at

- :doc:`../user_guide/config_settings`
- :doc:`../user_guide/data_intro`
- :doc:`../user_guide/model_intro`
- :doc:`../user_guide/evaluation_support`
- :doc:`../user_guide/usage`
