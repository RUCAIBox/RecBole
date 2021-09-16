Running Different Models
==========================
In RecBole, we have 4 categories of models, namely general recommendation, context-aware
recommendation, sequential recommendation and knowledge-based recommendation. Since different categories of models have different requirements for data
processing and evaluation setting, we need to configure these settings appropriately.

Here, we present some examples to show how to these four categories models in RecBole.


General Recommendation
---------------------------------

**specify and load the user and item columns**

General recommendation models utilize the historical interactions between
users and items to make recommendations, so it needs to specify and load the
user and item columns of the dataset.

.. code:: yaml

    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    load_col:
        inter: [user_id, item_id]

For some dataset, the column names corresponding to user and item in atomic
files may not be `user_id` and `item_id`. Just replace them with the
corresponding column names.

**training and evaluation settings**

General recommendation models usually need to group data by user and perform
negative sampling. You can set the config like this:

.. code:: yaml

    eval_args:
        group_by: user
    neg_sampling:
        uniform: 1

Context-aware Recommendation
------------------------------------

**load the feature columns**

Generally, context-aware recommendation models utilize the features of users, items and
interactions to make CTR predictions, so it needs to load the used features.

.. code:: yaml

    load_col:
        inter: [inter_feature1, inter_feature2]
        item: [item_feature1, item_feature2]
        user: [user_feature1, user_feature2]

`inter_feature1` refers to the column name of the corresponding feature in the
inter atomic file.

**label setting**

In general, context-aware recommendation models are mainly used in explicit feedback scenes, 
so your data should have explicit feedback information and you need to set label for them. For more information about label setting, 
please read the :doc:`../data/label_of_data`.

**evaluation settings**

If you want to apply context-aware recommendation models for CTR predictions, you can set the config like:

.. code:: yaml

    eval_args:
        group_by: None
        mode: labeled
    metrics: ['AUC', 'LogLoss']
    valid_metric: AUC

Note that RecBole also supports to evaluate the context-aware recommendation models by full-ranking like general recommendation models,
but you need to make sure that your ``.inter`` file can not load any other context information column.  
    

Sequential Recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**specify and load the user, item and time columns**

Sequential recommendation models utilize the historical interaction sequences
to predict hte next item, so it needs to specify and load the user, item and
time columns of the dataset.

.. code:: yaml

    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp
    load_col:
        inter: [user_id, item_id, timestamp]

For some dataset, the column names corresponding to user, item and time in
atomic files may not be `user_id`, `item_id` and `timestamp`, just replace them
with the corresponding column names.

**maximum length of the sequence**

The maximum length of the sequence can be modified by setting
``MAX_ITEM_LIST_LENGTH``

.. code:: yaml

    MAX_ITEM_LIST_LENGTH: 50

Knowledge-based Recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**specify and load the kg entity columns**

Knowledge-based recommendation models utilize KG information to make
recommendations, so it needs to specify and load the kg information of the dataset.

.. code:: yaml

    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    HEAD_ENTITY_ID_FIELD: head_id
    TAIL_ENTITY_ID_FIELD: tail_id
    RELATION_ID_FIELD: relation_id
    ENTITY_ID_FIELD: entity_id
    load_col:
        inter: [user_id, item_id]
        kg: [head_id, relation_id, tail_id]
        link: [item_id, entity_id]
