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

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.neg_sample_mixin import NegSampleMixin, NegSampleByMixin
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import DataLoaderType, InputType


class GeneralDataLoader(AbstractDataLoader):
    """:class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    dl_type = DataLoaderType.ORIGIN

    def __init__(self, config, dataset, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data


class GeneralNegSampleDataLoader(NegSampleByMixin, AbstractDataLoader):
    """:class:`GeneralNegSampleDataLoader` is a general-dataloader with negative sampling.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch. Beside this, when it is in the evaluation stage, and evaluator is topk-like function,
    we also permit that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        neg_sample_args (dict): The neg_sample_args of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(
        self, config, dataset, sampler, neg_sample_args, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False
    ):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.uid_list, self.uid2index, self.uid2items_num = None, None, None

        super().__init__(
            config, dataset, sampler, neg_sample_args, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle
        )

    def setup(self):
        if self.user_inter_in_one_batch:
            uid_field = self.dataset.uid_field
            user_num = self.dataset.user_num
            self.dataset.sort(by=uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(self.dataset.inter_feat[uid_field].numpy()):
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
        self._batch_size_adaptation()

    def _batch_size_adaptation(self):
        if self.user_inter_in_one_batch:
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > self.batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.upgrade_batch_size(new_batch_size)
        else:
            batch_num = max(self.batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.upgrade_batch_size(new_batch_size)

    @property
    def pr_end(self):
        if self.user_inter_in_one_batch:
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        if self.user_inter_in_one_batch:
            np.random.shuffle(self.uid_list)
        else:
            self.dataset.shuffle()

    def _next_batch_data(self):
        if self.user_inter_in_one_batch:
            uid_list = self.uid_list[self.pr:self.pr + self.step]
            data_list = []
            for uid in uid_list:
                index = self.uid2index[uid]
                data_list.append(self._neg_sampling(self.dataset[index]))
            cur_data = cat_interactions(data_list)
            pos_len_list = self.uid2items_num[uid_list]
            user_len_list = pos_len_list * self.times
            cur_data.set_additional_info(list(pos_len_list), list(user_len_list))
            self.pr += self.step
            return cur_data
        else:
            cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
            self.pr += self.step
            return cur_data

    def _neg_sampling(self, inter_feat):
        uids = inter_feat[self.uid_field]
        neg_iids = self.sampler.sample_by_user_ids(uids, self.neg_sample_by)
        return self.sampling_func(inter_feat, neg_iids)

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_iids):
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_iids})
        neg_item_feat = self.dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_iids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_iids
        new_data = self.dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data

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


class GeneralFullDataLoader(NegSampleMixin, AbstractDataLoader):
    """:class:`GeneralFullDataLoader` is a general-dataloader with full sort. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        neg_sample_args (dict): The neg_sample_args of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    dl_type = DataLoaderType.FULL

    def __init__(
        self, config, dataset, sampler, neg_sample_args, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False
    ):
        if neg_sample_args['strategy'] != 'full':
            raise ValueError('neg_sample strategy in GeneralFullDataLoader() should be `full`')

        uid_field = dataset.uid_field
        iid_field = dataset.iid_field
        user_num = dataset.user_num
        self.uid_list = []
        self.uid2items_num = np.zeros(user_num, dtype=np.int64)
        self.uid2swap_idx = np.array([None] * user_num)
        self.uid2rev_swap_idx = np.array([None] * user_num)
        self.uid2history_item = np.array([None] * user_num)

        dataset.sort(by=uid_field, ascending=True)
        last_uid = None
        positive_item = set()
        uid2used_item = sampler.used_ids
        for uid, iid in zip(dataset.inter_feat[uid_field].numpy(), dataset.inter_feat[iid_field].numpy()):
            if uid != last_uid:
                self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                last_uid = uid
                self.uid_list.append(uid)
                positive_item = set()
            positive_item.add(iid)
        self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
        self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
        self.user_df = dataset.join(Interaction({uid_field: self.uid_list}))

        super().__init__(
            config, dataset, sampler, neg_sample_args, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle
        )

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

    def _batch_size_adaptation(self):
        batch_num = max(self.batch_size // self.dataset.item_num, 1)
        new_batch_size = batch_num * self.dataset.item_num
        self.step = batch_num
        self.upgrade_batch_size(new_batch_size)

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _shuffle(self):
        self.logger.warnning('GeneralFullDataLoader can\'t shuffle')

    def _next_batch_data(self):
        user_df = self.user_df[self.pr:self.pr + self.step]
        cur_data = self._neg_sampling(user_df)
        self.pr += self.step
        return cur_data

    def _neg_sampling(self, user_df):
        uid_list = list(user_df[self.dataset.uid_field])
        pos_len_list = self.uid2items_num[uid_list]
        user_len_list = np.full(len(uid_list), self.item_num)
        user_df.set_additional_info(pos_len_list, user_len_list)

        history_item = self.uid2history_item[uid_list]
        history_row = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
        history_col = torch.cat(list(history_item))

        swap_idx = self.uid2swap_idx[uid_list]
        rev_swap_idx = self.uid2rev_swap_idx[uid_list]
        swap_row = torch.cat([torch.full_like(swap, i) for i, swap in enumerate(swap_idx)])
        swap_col_after = torch.cat(list(swap_idx))
        swap_col_before = torch.cat(list(rev_swap_idx))
        return user_df, (history_row, history_col), swap_row, swap_col_after, swap_col_before

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
        return np.full(self.pr_end, self.item_num)
