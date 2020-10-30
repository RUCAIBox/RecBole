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
import pandas as pd
import torch
from tqdm import tqdm

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.neg_sample_mixin import NegSampleMixin, NegSampleByMixin
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

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        super().__init__(config, dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        return self._dataframe_to_interaction(cur_data)


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
    def __init__(self, config, dataset, sampler, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.uid2index, self.uid2items_num = None, None

        super().__init__(config, dataset, sampler, neg_sample_args,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def setup(self):
        if self.user_inter_in_one_batch:
            self.uid2index, self.uid2items_num = self.dataset.uid2index
        self._batch_size_adaptation()

    def data_preprocess(self):
        if self.user_inter_in_one_batch:
            new_inter_num = 0
            new_inter_feat = []
            new_uid2index = []
            for uid, index in self.uid2index:
                new_inter_feat.append(self._neg_sampling(self.dataset.inter_feat[index]))
                new_num = len(new_inter_feat[-1])
                new_uid2index.append((uid, slice(new_inter_num, new_inter_num + new_num)))
                new_inter_num += new_num
            self.dataset.inter_feat = pd.concat(new_inter_feat, ignore_index=True)
            self.uid2index = np.array(new_uid2index)
        else:
            self.dataset.inter_feat = self._neg_sampling(self.dataset.inter_feat)

    def _batch_size_adaptation(self):
        if self.user_inter_in_one_batch:
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > self.batch_size:
                    break
                batch_num = i
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            batch_num = max(self.batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num if self.real_time else new_batch_size
            self.set_batch_size(new_batch_size)

    @property
    def pr_end(self):
        if self.user_inter_in_one_batch:
            return len(self.uid2index)
        else:
            return len(self.dataset)

    def _shuffle(self):
        if self.user_inter_in_one_batch:
            new_index = np.random.permutation(len(self.uid2index))
            self.uid2index = self.uid2index[new_index]
            self.uid2items_num = self.uid2items_num[new_index]
        else:
            self.dataset.shuffle()

    def _next_batch_data(self):
        if self.user_inter_in_one_batch:
            sampling_func = self._neg_sampling if self.real_time else (lambda x: x)
            cur_data = []
            for uid, index in self.uid2index[self.pr: self.pr + self.step]:
                cur_data.append(sampling_func(self.dataset[index]))
            cur_data = pd.concat(cur_data, ignore_index=True)
            pos_len_list = self.uid2items_num[self.pr: self.pr + self.step]
            user_len_list = pos_len_list * self.times
            self.pr += self.step
            return self._dataframe_to_interaction(cur_data, list(pos_len_list), list(user_len_list))
        else:
            cur_data = self.dataset[self.pr: self.pr + self.step]
            self.pr += self.step
            if self.real_time:
                cur_data = self._neg_sampling(cur_data)
            return self._dataframe_to_interaction(cur_data)

    def _neg_sampling(self, inter_feat):
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        uids = inter_feat[uid_field].to_list()
        neg_iids = self.sampler.sample_by_user_ids(uids, self.neg_sample_by)
        return self.sampling_func(uid_field, iid_field, neg_iids, inter_feat)

    def _neg_sample_by_pair_wise_sampling(self, uid_field, iid_field, neg_iids, inter_feat):
        inter_feat.insert(len(inter_feat.columns), self.neg_item_id, neg_iids)

        if self.dataset.item_feat is not None:
            neg_prefix = self.config['NEG_PREFIX']
            neg_item_feat = self.dataset.item_feat.add_prefix(neg_prefix)
            inter_feat = pd.merge(inter_feat, neg_item_feat,
                                  on=self.neg_item_id, how='left', suffixes=('_inter', '_item'))

        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, uid_field, iid_field, neg_iids, inter_feat):
        pos_inter_num = len(inter_feat)

        new_df = pd.concat([inter_feat] * self.times, ignore_index=True)
        new_df[iid_field].values[pos_inter_num:] = neg_iids

        labels = np.zeros(pos_inter_num * self.times, dtype=np.int64)
        labels[: pos_inter_num] = 1
        new_df[self.label_field] = labels

        return new_df

    def get_pos_len_list(self):
        """
        Returns:
            np.ndarray or list: Number of positive item for each user in a training/evaluating epoch.
        """
        return self.uid2items_num


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

    def __init__(self, config, dataset, sampler, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        if neg_sample_args['strategy'] != 'full':
            raise ValueError('neg_sample strategy in GeneralFullDataLoader() should be `full`')
        self.uid2index, self.uid2items_num = dataset.uid2index

        super().__init__(config, dataset, sampler, neg_sample_args,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def data_preprocess(self):
        self.user_tensor, tmp_pos_idx, tmp_used_idx, self.pos_len_list, self.neg_len_list = \
            self._neg_sampling(self.uid2index, show_progress=True)
        tmp_pos_len_list = [sum(self.pos_len_list[_: _ + self.step]) for _ in range(0, self.pr_end, self.step)]
        tot_item_num = self.dataset.item_num
        tmp_used_len_list = [sum(
            [tot_item_num - x for x in self.neg_len_list[_: _ + self.step]]
        ) for _ in range(0, self.pr_end, self.step)]
        self.pos_idx = list(torch.split(tmp_pos_idx, tmp_pos_len_list))
        self.used_idx = list(torch.split(tmp_used_idx, tmp_used_len_list))
        for i in range(len(self.pos_idx)):
            self.pos_idx[i] -= i * tot_item_num * self.step
        for i in range(len(self.used_idx)):
            self.used_idx[i] -= i * tot_item_num * self.step

    def _batch_size_adaptation(self):
        batch_num = max(self.batch_size // self.dataset.item_num, 1)
        new_batch_size = batch_num * self.dataset.item_num
        self.step = batch_num
        self.set_batch_size(new_batch_size)

    @property
    def pr_end(self):
        return len(self.uid2index)

    def _shuffle(self):
        self.logger.warnning('GeneralFullDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if not self.real_time:
            slc = slice(self.pr, self.pr + self.step)
            idx = self.pr // self.step
            cur_data = self.user_tensor[slc], self.pos_idx[idx], self.used_idx[idx], \
                self.pos_len_list[slc], self.neg_len_list[slc]
        else:
            cur_data = self._neg_sampling(self.uid2index[self.pr: self.pr + self.step])
        self.pr += self.step
        return cur_data

    def _neg_sampling(self, uid2index, show_progress=False):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        tot_item_num = self.dataset.item_num

        start_idx = 0
        pos_len_list = []
        neg_len_list = []

        pos_idx = []
        used_idx = []

        iter_data = tqdm(uid2index) if show_progress else uid2index
        for uid, index in iter_data:
            pos_item_id = self.dataset.inter_feat[iid_field][index].values
            pos_idx.extend([_ + start_idx for _ in pos_item_id])
            pos_num = len(pos_item_id)
            pos_len_list.append(pos_num)

            used_item_id = self.sampler.used_ids[uid]
            used_idx.extend([_ + start_idx for _ in used_item_id])
            used_num = len(used_item_id)

            neg_num = tot_item_num - used_num
            neg_len_list.append(neg_num)

            start_idx += tot_item_num

        user_df = pd.DataFrame({uid_field: np.array(uid2index[:, 0], dtype=np.int)})
        user_interaction = self._dataframe_to_interaction(self.join(user_df))

        return user_interaction, \
            torch.LongTensor(pos_idx), torch.LongTensor(used_idx), \
            pos_len_list, neg_len_list

    def get_pos_len_list(self):
        """
        Returns:
            np.ndarray or list: Number of positive item for each user in a training/evaluating epoch.
        """
        return self.uid2items_num
