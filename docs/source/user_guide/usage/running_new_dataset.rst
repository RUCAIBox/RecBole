Running New Dataset
=======================
RecBole has a build-in dataset **ml-100k** for users to quickly get start. 
However, if you want to use new dataset, here, we present how to use a new dataset in RecBole.


Prepare atomic files
-------------------------

In order to characterize most forms of the input data required by different recommendation tasks, 
RecBole designs an input data format called :doc:`../data/atomic_files` and 
you need to convert your raw data into Atomic Files format before data loading. 

For the convenience of users, we have collected more than
28 commonly used datasets (detailed as `Dataset List </dataset_list.html>`_.) and released their Atomic Files format 
for users to download them freely. More information of downloading our prepared datasets can be found in :doc:`../data/dataset_download`.

However, if you use other datasets, you should convert your data into the Atomic Files by yourself.

For the ml-1m dataset, the converted atomic files are like:

**ml-1m.inter**

=============   =============   ============   ===============
user_id:token   item_id:token   rating:float   timestamp:float
=============   =============   ============   ===============
1               1193            5              978300760
1               661             3              978302109
=============   =============   ============   ===============

**ml-1m.user**

=============   =========   ============   ================   ==============
user_id:token   age:token   gender:token   occupation:token   zip_code:token
=============   =========   ============   ================   ==============
1               1           F              10                 48067
2               56          M              16                 70072
=============   =========   ============   ================   ==============

**ml-1m.item**

=============   =====================   ==================   ============================
item_id:token   movie_title:token_seq   release_year:token   genre:token_seq
=============   =====================   ==================   ============================
1               Toy Story               1995                 Animation Children's Comedy
2               Jumanji                 1995                 Adventure Children's Fantasy
=============   =====================   ==================   ============================


Set data path
---------------
You need to set the data path in config when you want to use new dataset. 
The name of atomic files, name of dir that containing atomic files and ``config['dataset']`` should be the same, and
the ``data_path`` in your config should be the parent dir of the directory that contains atomic files.

For example:

.. code:: none

    ~/xxx/yyy/ml-1m/
    ├── ml-1m.inter
    ├── ml-1m.item
    ├── ml-1m.kg
    ├── ml-1m.link
    └── ml-1m.user

.. code:: yaml

    data_path: ~/xxx/yyy/
    dataset: ml-1m

Convert to Dataset
---------------------
Here, we present how to convert atomic files into :class:`~recbole.data.dataset.dataset.Dataset`.

Suppose we use ml-1m to train BPR.

According to the dataset information, the user should set the dataset information and filtering parameters in the configuration file `ml-1m.yaml`.
For example, we conduct 10-core filtering, removing the ratings which are smaller than 3, the time of the record should be earlier than 97830000, and we only load inter data.
The ``yaml`` file should be like:

.. code:: yaml

    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    RATING_FIELD: rating
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, rating, timestamp]

    user_inter_num_interval: "[10,inf)"
    item_inter_num_interval: "[10,inf)"
    val_interval:
        rating: "[3,inf)"
        timestamp: "[97830000, inf)"


.. code:: python

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation

    if __name__ == '__main__':
        config = Config(model='BPR', dataset='ml-1m', config_file_list=['ml-1m.yaml'])
        dataset = create_dataset(config)


Convert to Dataloader
------------------------
Here, we present how to convert :class:`~recbole.data.dataset.dataset.Dataset` into :obj:`Dataloader`.

We firstly set the parameters in the configuration file `ml-1m.yaml`.
Suppose we want to leverage random ordering, ratio-based splitting and full ranking with all item candidates, the splitting ratio is set as 8:1:1.
You can add the following config in your `ml-1m.yaml`:

.. code:: yaml

    eval_args:
        split: {'RS': [8,1,1]}
        group_by: user
        order: RO
        mode: full


.. code:: python

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation


    if __name__ == '__main__':

        ...

        train_data, valid_data, test_data = data_preparation(config, dataset)
