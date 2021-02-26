Running Different Models
==========================
Here, we present how to run different models in RecBole.

Proper Parameters Configuration
----------------------------------
Since different categories of models have different requirements for data
processing and evaluation setting, we need to configure these settings
appropriately.

The following will introduce the parameter configuration of these four
categories of models: namely general recommendation, context-aware
recommendation, sequential recommendation and knowledge-based recommendation.

General Recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^^

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

General recommendation models usually needs to group data by user and perform
negative sampling.

.. code:: yaml

    group_by_user: True
    training_neg_sample_num: 1

Context-aware Recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**load the feature columns**

Context-aware recommendation models utilize the features of users, items and
interactions to make CTR predictions, so it needs to load the used features.

.. code:: yaml

    load_col:
        inter: [inter_feature1, inter_feature2]
        item: [item_feature1, item_feature2]
        user: [user_feature1, user_feature2]

`inter_feature1` refers to the column name of the corresponding feature in the
inter atomic file.

**label setting**

We also need to configure `LABEL_FIELD`, which represents the label column in
the CTR prediction. For the Context-aware recommendation models, the setting of
`LABEL_FIELD` is divided into two cases:

1) There is a label field in atomic file, and the value is in 0/1, we only need to
set as follows:

.. code:: yaml

    LABEL_FIELD: label

2) There is no label field in atomic file, we need to generate label field based
on some information.

.. code:: yaml

    LABEL_FIELD: label
    threshold:
        rating: 3

`rating` is a column in atomic file and is loaded (by ``load_col``). In this way,
the label of the interaction with ``rating >= 3`` is set to 1, the reset are
set to 0.

**training and evaluation settings**

Context-aware recommendation models usually does not need to group data by user and
perform negative sampling.

.. code:: yaml

    group_by_user: False
    training_neg_sample_num: 0

Since there is no need to rank the results, ``eval_setting`` only needs to set
the first part, for example:

.. code:: yaml

    eval_setting: RO_RS

The evaluation metrics are generally set to `AUC` and `LogLoss`.

.. code:: yaml

    metrics: ['AUC', 'LogLoss']


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
