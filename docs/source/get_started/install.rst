Install RecBole
======================
RecBole can be installed from ``conda``, ``pip`` and source files.


System requirements
------------------------
RecBole is compatible with the following operating systems:

* Linux
* Windows 10
* macOS X

Python 3.7 (or later), torch 1.7.0 (or later) are required to install our library. If you want to use RecBole with GPU,
please ensure that CUDA or CUDAToolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).


Install from conda
--------------------------
``Conda`` can be installed from `miniconda <https://conda.io/miniconda.html>`_ or
the full `anaconda <https://www.anaconda.com/download/>`_.
If you are in China, `Tsinghua Mirrors <https://mirror.tuna.tsinghua.edu.cn/help/anaconda/>`_ is recommended.

After installing ``conda``,
run `conda create -n recbole python=3.7` to create the Python 3.7 conda environment.
Then the environment can be activated by `conda activate recbole`.
At last, run the following command to install RecBole:

.. code:: bash

    conda install -c aibox recbole


Install from pip
-------------------------
To install RecBole from pip, only the following command is needed:

.. code:: bash

    pip install recbole


Install from source
-------------------------
Download the source files from GitHub.

.. code:: bash

    git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole

Run the following command to install:

.. code:: bash

    pip install -e . --verbose

Try to run：
-------------------------
To check if you have successfully installed the RecBole, you can create a new python file (e.g., `run.py`),
and write the following code:

.. code:: python

    from recbole.quick_start import run_recbole

    run_recbole(model='BPR', dataset='ml-100k')


Then run the following command:

.. code:: bash

    python run.py

This will perform the training and test of the BPR model on the ml-100k dataset, and you will obtain some output like:

.. code:: none

    05 Aug 02:16    INFO  ml-100k
    The number of users: 944
    Average actions of users: 106.04453870625663
    The number of items: 1683
    Average actions of items: 59.45303210463734
    The number of inters: 100000
    The sparsity of the dataset: 93.70575143257098%
    Remain Fields: ['user_id', 'item_id', 'rating', 'timestamp']
    05 Aug 02:16    INFO  [Training]: train_batch_size = [2048] negative sampling: [{'uniform': 1}]
    05 Aug 02:16    INFO  [Evaluation]: eval_batch_size = [4096] eval_args: [{'split': {'RS': [0.8, 0.1, 0.1]}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'}]
    05 Aug 02:16    INFO  BPR(
    (user_embedding): Embedding(944, 64)
    (item_embedding): Embedding(1683, 64)
    (loss): BPRLoss()
    )
    Trainable parameters: 168128
    Train     0: 100%|████████████████████████| 40/40 [00:00<00:00, 219.54it/s, GPU RAM: 0.01 G/11.91 G]
    05 Aug 02:16    INFO  epoch 0 training [time: 0.19s, train loss: 27.7228]
    Evaluate   : 100%|██████████████████████| 472/472 [00:00<00:00, 506.11it/s, GPU RAM: 0.01 G/11.91 G]
    05 Aug 02:16    INFO  epoch 0 evaluating [time: 0.94s, valid_score: 0.020500]
    05 Aug 02:16    INFO  valid result: 
    recall@10 : 0.0067    mrr@10 : 0.0205    ndcg@10 : 0.0086    hit@10 : 0.0732    precision@10 : 0.0081    

    ...

    Train    96: 100%|████████████████████████| 40/40 [00:00<00:00, 230.65it/s, GPU RAM: 0.01 G/11.91 G]
    05 Aug 02:19    INFO  epoch 96 training [time: 0.18s, train loss: 3.7170]
    Evaluate   : 100%|██████████████████████| 472/472 [00:00<00:00, 800.46it/s, GPU RAM: 0.01 G/11.91 G]
    05 Aug 02:19    INFO  epoch 96 evaluating [time: 0.60s, valid_score: 0.375200]
    05 Aug 02:19    INFO  valid result: 
    recall@10 : 0.2162    mrr@10 : 0.3752    ndcg@10 : 0.2284    hit@10 : 0.7508    precision@10 : 0.1602    
    05 Aug 02:19    INFO  Finished training, best eval result in epoch 85
    05 Aug 02:19    INFO  Loading model structure and parameters from saved/BPR-Aug-05-2021_02-17-51.pth
    Evaluate   : 100%|██████████████████████| 472/472 [00:00<00:00, 832.85it/s, GPU RAM: 0.01 G/11.91 G]
    05 Aug 02:19    INFO  best valid : {'recall@10': 0.2195, 'mrr@10': 0.3871, 'ndcg@10': 0.2344, 'hit@10': 0.7582, 'precision@10': 0.1627}
    05 Aug 02:19    INFO  test result: {'recall@10': 0.2523, 'mrr@10': 0.4855, 'ndcg@10': 0.292, 'hit@10': 0.7953, 'precision@10': 0.1962}