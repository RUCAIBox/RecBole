# @Time   : 2021/03/20
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn


"""
Case study example
===================
Here is the sample code for the case study in RecBole.
"""


import torch
from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.quick_start import load_data_and_model


if __name__ == '__main__':
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file='../saved/BPR-Aug-20-2021_03-32-13.pth',
    )  # Here you can replace it by your model path.

    # uid_series = np.array([1, 2])  # internal user id series
    # or you can use dataset.token2id to transfer external user token to internal user id
    uid_series = dataset.token2id(dataset.uid_field, ['196', '186'])

    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
    print(topk_score)  # scores of top 10 items
    print(topk_iid_list)  # internal id of top 10 items
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    print(external_item_list)  # external tokens of top 10 items
    print()

    score = full_sort_scores(uid_series, model, test_data, device=config['device'])
    print(score)  # score of all items
    print(score[0, dataset.token2id(dataset.iid_field, ['242', '302'])])  # score of item ['242', '302'] for user '196'.
