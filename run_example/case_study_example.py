# @Time   : 2021/03/20
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn


"""
Case study example
===================
Here is the sample code for the case study in RecBole.
"""


import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, init_seed
from recbole.utils.case_study import full_sort_topk, full_sort_scores


if __name__ == '__main__':
    # this part is to load saved model.
    config_dict = {
        # here you can set some parameters such as `gpu_id` and so on.
    }
    config = Config(model='BPR', dataset='ml-100k', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # Here you can also use `load_split_dataloaders` to load data.
    # The example code for `load_split_dataloaders` can be found in `save_and_load_example.py`.

    model = get_model(config['model'])(config, train_data)
    checkpoint = torch.load('RecBole/saved/BPR-Dec-08-2020_15-37-37.pth')  # Here you can replace it by your model path.
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # uid_series = np.array([1, 2])  # internal user id series
    # or you can use dataset.token2id to transfer external user token to internal user id
    uid_series = dataset.token2id(dataset.uid_field, ['200'])

    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10)
    print(topk_score)  # scores of top 10 items
    print(topk_iid_list)  # internal id of top 10 items
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list)
    print(external_item_list)  # external tokens of top 10 items
    print()

    score = full_sort_scores(uid_series, model, test_data)
    print(score)  # score of all items
    print(score[0, dataset.token2id(dataset.iid_field, ['242', '302'])])  # score of item ['242', '302'] for user '196'.
