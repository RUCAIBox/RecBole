Case study
=============

Case study is an in-depth study of the performance of a specific recommendation algorithm,
which will analysis the recommendation result of some users.
In RecBole, we implemented :meth:`~recbole.utils.case_study.full_sort_scores`
and :meth:`~recbole.utils.case_study.full_sort_topk` for case study purpose.
In this section, we will present a typical usage of these two functions.

Reload model
-------------

First, we need to reload the recommendation model,
we can use :meth:`~recbole.quick_start.quick_start.load_data_and_model` to load saved data and model.

.. code:: python3

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file='../saved/BPR-Aug-20-2021_03-32-13.pth',
    )  # Here you can replace it by your model path.

Convert external user id into internal user id
-------------------------------------------------

Then, we need to use :meth:`~recbole.data.dataset.dataset.Dataset.token2id`
to convert external user id which we want to do case study into internal user id.

.. code:: python3

    uid_series = dataset.token2id(dataset.uid_field, ['196', '186'])

Get scores of every user-item pairs
-------------------------------------

If we want to calculate the scores of every user-item pairs for given user,
we can call :meth:`~recbole.utils.case_study.full_sort_scores` function to get the scores matrix.

.. code:: python3

    score = full_sort_scores(uid_series, model, test_data, device=config['device'])
    print(score)  # score of all items
    print(score[0, dataset.token2id(dataset.iid_field, ['242', '302'])])
    # score of item ['242', '302'] for user '196'.

The output will be like this:

.. code:: none

    tensor([[   -inf,    -inf,  0.1074,  ..., -0.0966, -0.1217, -0.0966],
            [   -inf, -0.0013,    -inf,  ..., -0.1115, -0.1089, -0.1196]],
           device='cuda:0')
    tensor([  -inf, 0.1074], device='cuda:0')

Note that the score of ``[pad]`` and history items (for non-repeatable recommendation) will be set into ``-inf``.

Get the top ranked item for each user
--------------------------------------

If we want to get the top ranked item for given user,
we can call :meth:`~recbole.utils.case_study.full_sort_topk` function to get the scores and internal ids of these items.

.. code:: python3

    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
    print(topk_score)  # scores of top 10 items
    print(topk_iid_list)  # internal id of top 10 items
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    print(external_item_list)  # external tokens of top 10 items

The output will be like this:

.. code:: none

    tensor([[0.1985, 0.1947, 0.1850, 0.1849, 0.1822, 0.1770, 0.1770, 0.1765, 0.1752,
             0.1744],
            [0.2487, 0.2379, 0.2351, 0.2311, 0.2293, 0.2239, 0.2215, 0.2156, 0.2137,
             0.2114]], device='cuda:0')
    tensor([[ 50,  32, 158, 210,  13, 100, 201,  61, 167, 312],
            [102, 312, 358, 100,  32,  53, 167, 472, 162, 201]], device='cuda:0')
    [['100' '98' '258' '7' '222' '496' '318' '288' '216' '176']
     ['174' '176' '50' '496' '98' '181' '216' '28' '172' '318']]
