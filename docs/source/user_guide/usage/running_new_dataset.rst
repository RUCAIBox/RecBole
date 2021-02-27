Running New Dataset
=======================
Here, we present how to use a new dataset in RecBole.


Convert to Atomic Files
-------------------------

If the user use the collected datasets, she can choose one of the following ways:

1. Download the converted atomic files from `Google Drive <https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj?usp=sharing>`_ or `Baidu Wangpan <https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g>`_ (Password: e272).
2. Find the converting script from RecDatasets_, and transform them to atomic files.

If the user use other datasets, she should format the data according to the format of the atomic files.

.. _RecDatasets: https://github.com/RUCAIBox/RecDatasets

For the dataset of ml-1m, the converting file is:

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


Local Path
---------------
Name of atomic files, name of dir that containing atomic files and ``config['dataset']`` should be the same.

``config['data_path']`` should be the parent dir of the dir that containing atomic files.

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

.. code:: yaml

    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    RATING_FIELD: rating
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, rating, timestamp]

    min_user_inter_num: 10
    min_item_inter_num: 10
    lowest_val:
        rating: 3
        timestamp: 97830000


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
We leverage random ordering + ratio-based splitting and full ranking with all item candidates, the splitting ratio is set as 8:1:1.

.. code:: yaml

    ...

    eval_setting: RO_RS,full
    split_ratio: [0.8,0.1,0.1]


.. code:: python

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation


    if __name__ == '__main__':

        ...

        train_data, valid_data, test_data = data_preparation(config, dataset)
