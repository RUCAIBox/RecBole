# @Time   : 2020/12/25
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time   : 2020/12/25
# @Author : Yushuo Chen
# @email  : chenyushuo@ruc.edu.cn

"""
recbole.utils.case_study
#####################################
"""

import numpy as np
import torch

from recbole.data.dataloader.general_dataloader import GeneralFullDataLoader
from recbole.data.dataloader.sequential_dataloader import SequentialFullDataLoader


@torch.no_grad()
def full_sort_scores(uid_series, model, test_data):
    """Calculate the scores of all items for each user in uid_series.

    Note:
        The score of [pad] and history items will be set into -inf.

    Args:
        uid_series (numpy.ndarray): User id series
        model (AbstractRecommender): Model to predict
        test_data (AbstractDataLoader): The test_data of model

    Returns:
        torch.Tensor: the scores of all items for each user in uid_series.
    """
    uid_field = test_data.dataset.uid_field
    dataset = test_data.dataset
    model.eval()

    if isinstance(test_data, GeneralFullDataLoader):
        index = np.isin(test_data.user_df[uid_field].numpy(), uid_series)
        input_interaction = test_data.user_df[index]
        history_item = test_data.uid2history_item[input_interaction[uid_field].numpy()]
        history_row = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
        history_col = torch.cat(list(history_item))
        history_index = history_row, history_col
    elif isinstance(test_data, SequentialFullDataLoader):
        index = np.isin(test_data.uid_list, uid_series)
        input_interaction = test_data.augmentation(
            test_data.item_list_index[index], test_data.target_index[index], test_data.item_list_length[index]
        )
        history_index = None
    else:
        raise NotImplementedError

    # Get scores of all items
    try:
        scores = model.full_sort_predict(input_interaction)
    except NotImplementedError:
        input_interaction = input_interaction.repeat(dataset.item_num)
        input_interaction.update(test_data.get_item_feature().repeat(len(uid_series)))
        scores = model.predict(input_interaction)

    scores = scores.view(-1, dataset.item_num)
    scores[:, 0] = -np.inf  # set scores of [pad] to -inf
    if history_index is not None:
        scores[history_index] = -np.inf  # set scores of history items to -inf

    return scores


def full_sort_topk(uid_series, model, test_data, k):
    """Calculate the top-k items' scores and ids for each user in uid_series.

    Args:
        uid_series (numpy.ndarray): User id series
        model (AbstractRecommender): Model to predict
        test_data (AbstractDataLoader): The test_data of model
        k (int): The top-k items.

    Returns:
        tuple:
            - topk_scores (torch.Tensor): The scores of topk items.
            - topk_index (torch.Tensor): The index of topk items, which is also the internal ids of items.
    """
    scores = full_sort_scores(uid_series, model, test_data)
    return torch.topk(scores, k)
