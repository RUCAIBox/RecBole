# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/29
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader, NegSampleDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import DataLoaderType, InputType, ModelType


class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self._set_neg_sample_args(config, dataset, config['MODEL_INPUT_TYPE'], config['train_neg_sample_args'])
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        if self.neg_sample_args['strategy'] == 'by':
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data


class NegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    def __init__(self, config, dataset, sampler, shuffle=False):
        user_num = dataset.user_num
        dataset.sort(by=dataset.uid_field, ascending=True)
        self.uid_list = []
        start, end = dict(), dict()
        for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
            if uid not in start:
                self.uid_list.append(uid)
                start[uid] = i
            end[uid] = i
        self.uid2index = np.array([None] * user_num)
        self.uid2items_num = np.zeros(user_num, dtype=np.int64)
        for uid in self.uid_list:
            self.uid2index[uid] = slice(start[uid], end[uid] + 1)
            self.uid2items_num[uid] = end[uid] - start[uid] + 1
        self.uid_list = np.array(self.uid_list)

        self._set_neg_sample_args(config, dataset, InputType.POINTWISE, config['eval_neg_sample_args'])
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['eval_batch_size']
        if self.neg_sample_args['strategy'] == 'by':
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _shuffle(self):
        self.logger.warnning('NegSampleEvalDataLoader can\'t shuffle')

    def _next_batch_data(self):
        uid_list = self.uid_list[self.pr:self.pr + self.step]
        data_list = []
        for uid in uid_list:
            index = self.uid2index[uid]
            data_list.append(self._neg_sampling(self.dataset[index]))
        cur_data = cat_interactions(data_list)
        if self.neg_sample_args['strategy'] == 'by':
            pos_len_list = self.uid2items_num[uid_list]
            user_len_list = pos_len_list * self.times
            cur_data.set_additional_info(list(pos_len_list), list(user_len_list))
        self.pr += self.step
        return cur_data

    def get_pos_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of positive item for each user in a training/evaluating epoch.
        """
        return self.uid2items_num[self.uid_list]

    def get_user_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of all item for each user in a training/evaluating epoch.
        """
        return self.uid2items_num[self.uid_list] * self.times


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    dl_type = DataLoaderType.FULL

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config['MODEL_TYPE'] == ModelType.SEQUENTIAL
        if not self.is_sequential:
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            self.uid2swap_idx = np.array([None] * user_num)
            self.uid2rev_swap_idx = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)
            last_uid = None
            positive_item = set()
            uid2used_item = sampler.used_ids
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                if uid != last_uid:
                    self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                positive_item.add(iid)
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        positive_item_num = len(positive_item)
        self.uid2items_num[uid] = positive_item_num
        swap_idx = torch.tensor(sorted(set(range(positive_item_num)) ^ positive_item))
        self.uid2swap_idx[uid] = swap_idx
        self.uid2rev_swap_idx[uid] = swap_idx.flip(0)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config['eval_batch_size']
        if not self.is_sequential:
            batch_num = max(batch_size // self.dataset.item_num, 1)
            new_batch_size = batch_num * self.dataset.item_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        if not self.is_sequential:
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        self.logger.warnning('FullSortEvalDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if not self.is_sequential:
            user_df = self.user_df[self.pr:self.pr + self.step]
            uid_list = list(user_df[self.dataset.uid_field])
            pos_len_list = self.uid2items_num[uid_list]
            user_len_list = np.full(len(uid_list), self.dataset.item_num)
            user_df.set_additional_info(pos_len_list, user_len_list)

            history_item = self.uid2history_item[uid_list]
            history_row = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_col = torch.cat(list(history_item))

            swap_idx = self.uid2swap_idx[uid_list]
            rev_swap_idx = self.uid2rev_swap_idx[uid_list]
            swap_row = torch.cat([torch.full_like(swap, i) for i, swap in enumerate(swap_idx)])
            swap_col_after = torch.cat(list(swap_idx))
            swap_col_before = torch.cat(list(rev_swap_idx))

            self.pr += self.step
            return user_df, (history_row, history_col), swap_row, swap_col_after, swap_col_before
        else:
            interaction = self.dataset[self.pr:self.pr + self.step]
            inter_num = len(interaction)
            pos_len_list = np.ones(inter_num, dtype=np.int64)
            user_len_list = np.full(inter_num, self.dataset.item_num)
            interaction.set_additional_info(pos_len_list, user_len_list)
            scores_row = torch.arange(inter_num).repeat(2)
            padding_idx = torch.zeros(inter_num, dtype=torch.int64)
            positive_idx = interaction[self.iid_field]
            scores_col_after = torch.cat((padding_idx, positive_idx))
            scores_col_before = torch.cat((positive_idx, padding_idx))

            self.pr += self.step
            return interaction, None, scores_row, scores_col_after, scores_col_before

    def get_pos_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of positive item for each user in a training/evaluating epoch.
        """
        if not self.is_sequential:
            return self.uid2items_num[self.uid_list]
        else:
            return np.ones(self.pr_end, dtype=np.int64)

    def get_user_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of all item for each user in a training/evaluating epoch.
        """
        return np.full(self.pr_end, self.dataset.item_num)
