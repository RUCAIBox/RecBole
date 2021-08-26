Save and load data and model
==============================

In this section, we will present how to save and load data and model.

Save data and model
--------------------

When we use the :meth:`~recbole.quick_start.quick_start.run_recbole` function mentioned in :doc:`run_recbole`,
it will save the best model parameters in training process and its corresponding config settings.
If you want to save filtered dataset and split dataloaders,
you can set parameter :attr:`save_dataset` and parameter :attr:`save_dataloaders` to ``True``
to save filtered dataset and split dataloaders.

You can refer to :doc:`../config_settings` for more details about :attr:`save_dataset` and :attr:`save_dataloaders`.

Here we present a typical output when two parameters above is ``True``:

.. code:: none

    21 Aug 13:05    INFO  Saving filtered dataset into [saved/ml-100k-dataset.pth]
    21 Aug 13:05    INFO  ml-100k
    The number of users: 944
    Average actions of users: 106.04453870625663
    The number of items: 1683
    Average actions of items: 59.45303210463734
    The number of inters: 100000
    The sparsity of the dataset: 93.70575143257098%
    Remain Fields: ['user_id', 'item_id', 'rating', 'timestamp']
    21 Aug 13:05    INFO  Saved split dataloaders: saved/ml-100k-for-BPR-dataloader.pth
    21 Aug 13:06    INFO  BPR(
        (user_embedding): Embedding(944, 64)
        (item_embedding): Embedding(1683, 64)
        (loss): BPRLoss()
    )
    Trainable parameters: 168128
    Train     0: 100%|█████████████████████████| 40/40 [00:01<00:00, 32.52it/s, GPU RAM: 0.01 G/11.91 G]
    21 Aug 13:06    INFO  epoch 0 training [time: 1.24s, train loss: 27.7228]
    Evaluate   : 100%|███████████████████████| 472/472 [00:04<00:00, 94.53it/s, GPU RAM: 0.01 G/11.91 G]
    21 Aug 13:06    INFO  epoch 0 evaluating [time: 5.00s, valid_score: 0.020500]
    21 Aug 13:06    INFO  valid result:
    recall@10 : 0.0067    mrr@10 : 0.0205    ndcg@10 : 0.0086    hit@10 : 0.0732    precision@10 : 0.0081
    21 Aug 13:06    INFO  Saving current best: saved/BPR-Aug-21-2021_13-06-00.pth

    ...

As we can see, the filtered dataset is saved to ``saved/ml-100k-dataset.pth``,
the split dataloaders are saved to ``saved/ml-100k-for-BPR-dataloader.pth``,
and the model is saved to ``saved/BPR-Aug-21-2021_13-06-00.pth``.

Load data and model
--------------------

If you want to reload the data and model,
you can apply :meth:`~recbole.quick_start.quick_start.load_data_and_model` to get them.
You can also pass :attr:`dataset_file` and :attr:`dataloader_file` to this function to reload data from file,
which can reduce the time of data filtering and data splitting.

Here we present a typical usage of :meth:`~recbole.quick_start.quick_start.load_data_and_model`:

.. code:: python3

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file='saved/BPR-Aug-21-2021_13-06-00.pth',
    )
    # Here you can replace it by your model path.
    # And you can also pass 'dataset_file' and 'dataloader_file' to this function.
