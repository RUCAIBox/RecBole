# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/6, 2020/9/17
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.dataloader.sequential_dataloader
################################################
"""

import numpy as np
import torch

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.neg_sample_mixin import NegSampleByMixin
from recbole.utils import DataLoaderType, FeatureSource, FeatureType, InputType


class SequentialDataLoader(AbstractDataLoader):
    """:class:`SequentialDataLoader` is used for sequential model. It will do data augmentation for the origin data.
    And its returned data contains the following:

        - user id
        - history items list
        - history items' interaction time list
        - item to be predicted
        - the interaction time of item to be predicted
        - history list length
        - other interaction information of item to be predicted

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
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.time_field = dataset.time_field
        self.max_item_list_len = config['MAX_ITEM_LIST_LENGTH']

        list_suffix = config['LIST_SUFFIX']
        self.item_list_field = self.iid_field + list_suffix
        self.time_list_field = self.time_field + list_suffix
        self.position_field = config['POSITION_FIELD']
        self.target_iid_field = self.iid_field
        self.target_time_field = self.time_field
        self.item_list_length_field = config['ITEM_LIST_LENGTH_FIELD']

        dataset.set_field_property(self.item_list_field, FeatureType.TOKEN_SEQ, FeatureSource.INTERACTION,
                                   self.max_item_list_len)
        dataset.set_field_property(self.time_list_field, FeatureType.FLOAT_SEQ, FeatureSource.INTERACTION,
                                   self.max_item_list_len)
        if self.position_field:
            dataset.set_field_property(self.position_field, FeatureType.TOKEN_SEQ, FeatureSource.INTERACTION,
                                       self.max_item_list_len)
        dataset.set_field_property(self.target_iid_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        dataset.set_field_property(self.target_time_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        dataset.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)

        self.uid_list, self.item_list_index, self.target_index, self.item_list_length = \
            dataset.prepare_data_augmentation()
        self.pre_processed_data = None

        super().__init__(config, dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def data_preprocess(self):
        """Do data augmentation before training/evaluation.
        """
        self.pre_processed_data = self.augmentation(self.uid_list, self.item_list_field,
                                                    self.target_index, self.item_list_length)

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _shuffle(self):
        new_index = np.random.permutation(len(self.item_list_index))
        if self.real_time:
            self.uid_list = self.uid_list[new_index]
            self.item_list_index = self.item_list_index[new_index]
            self.target_index = self.target_index[new_index]
            self.item_list_length = self.item_list_length[new_index]
        else:
            new_data = {}
            for key, value in self.pre_processed_data.items():
                new_data[key] = value[new_index]
            self.pre_processed_data = new_data

    def _next_batch_data(self):
        cur_index = slice(self.pr, self.pr + self.step)
        if self.real_time:
            cur_data = self.augmentation(self.uid_list[cur_index],
                                         self.item_list_index[cur_index],
                                         self.target_index[cur_index],
                                         self.item_list_length[cur_index])
        else:
            cur_data = {}
            for key, value in self.pre_processed_data.items():
                cur_data[key] = value[cur_index]
        self.pr += self.step
        return self._dict_to_interaction(cur_data)

    def augmentation(self, uid_list, item_list_index, target_index, item_list_length):
        """Data augmentation.

        Args:
            uid_list (np.ndarray): user id list.
            item_list_index (np.ndarray): the index of history items list in interaction.
            target_index (np.ndarray): the index of items to be predicted in interaction.
            item_list_length (np.ndarray): history list length.

        Returns:
            dict: the augmented data.
        """
        new_length = len(item_list_index)
        new_dict = {
            self.uid_field: uid_list,
            self.item_list_field: np.zeros((new_length, self.max_item_list_len), dtype=np.int64),
            self.time_list_field: np.zeros((new_length, self.max_item_list_len), dtype=np.int64),
            self.target_iid_field: self.dataset.inter_feat[self.iid_field][target_index].values,
            self.target_time_field: self.dataset.inter_feat[self.time_field][target_index].values,
            self.item_list_length_field: item_list_length,
        }
        for field in self.dataset.inter_feat:
            if field != self.iid_field and field != self.time_field:
                new_dict[field] = self.dataset.inter_feat[field][target_index].values
        if self.position_field:
            new_dict[self.position_field] = np.tile(np.arange(self.max_item_list_len), (new_length, 1))

        iid_value = self.dataset.inter_feat[self.iid_field].values
        time_value = self.dataset.inter_feat[self.time_field].values
        for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
            new_dict[self.item_list_field][i][:length] = iid_value[index]
            new_dict[self.time_list_field][i][:length] = time_value[index]
        return new_dict


class SequentialNegSampleDataLoader(NegSampleByMixin, SequentialDataLoader):
    """:class:`SequentialNegSampleDataLoader` is sequential-dataloader with negative sampling.
    Like :class:`~recbole.data.dataloader.general_dataloader.GeneralNegSampleDataLoader`, for the result of every batch,
    we permit that every positive interaction and its negative interaction must be in the same batch. Beside this,
    when it is in the evaluation stage, and evaluator is topk-like function, we also permit that all the interactions
    corresponding to each user are in the same batch and positive interactions are before negative interactions.

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
        super().__init__(config, dataset, sampler, neg_sample_args,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def data_preprocess(self):
        """Do data augmentation and neg-sampling before training/evaluation.
        """
        self.pre_processed_data = self.augmentation(self.uid_list, self.item_list_field,
                                                    self.target_index, self.item_list_length)
        self.pre_processed_data = self._neg_sampling(self.pre_processed_data)

    def _batch_size_adaptation(self):
        batch_num = max(self.batch_size // self.times, 1)
        new_batch_size = batch_num * self.times
        self.step = batch_num if self.real_time else new_batch_size
        self.set_batch_size(new_batch_size)

    def _next_batch_data(self):
        cur_index = slice(self.pr, self.pr + self.step)
        if self.real_time:
            cur_data = self.augmentation(self.uid_list[cur_index],
                                         self.item_list_index[cur_index],
                                         self.target_index[cur_index],
                                         self.item_list_length[cur_index])
            cur_data = self._neg_sampling(cur_data)
        else:
            cur_data = {}
            for key, value in self.pre_processed_data.items():
                cur_data[key] = value[cur_index]
        self.pr += self.step

        if self.user_inter_in_one_batch:
            cur_data_len = len(cur_data[self.uid_field])
            pos_len_list = np.ones(cur_data_len // self.times, dtype=np.int64)
            user_len_list = pos_len_list * self.times
            return self._dict_to_interaction(cur_data, list(pos_len_list), list(user_len_list))
        else:
            return self._dict_to_interaction(cur_data)

    def _neg_sampling(self, data):
        if self.user_inter_in_one_batch:
            data_len = len(data[self.uid_field])
            data_list = []
            for i in range(data_len):
                uids = data[self.uid_field][i: i + 1]
                neg_iids = self.sampler.sample_by_user_ids(uids, self.neg_sample_by)
                cur_data = {field: data[field][i: i + 1] for field in data}
                data_list.append(self.sampling_func(cur_data, neg_iids))
            return {field: np.concatenate([d[field] for d in data_list])
                    for field in data}
        else:
            uids = data[self.uid_field]
            neg_iids = self.sampler.sample_by_user_ids(uids, self.neg_sample_by)
            return self.sampling_func(data, neg_iids)

    def _neg_sample_by_pair_wise_sampling(self, data, neg_iids):
        data[self.neg_item_id] = neg_iids
        return data

    def _neg_sample_by_point_wise_sampling(self, data, neg_iids):
        new_data = {}
        for key, value in data.items():
            if key == self.target_iid_field:
                new_data[key] = np.concatenate([value, neg_iids])
            else:
                new_data[key] = np.concatenate([value] * self.times)
        pos_len = len(data[self.target_iid_field])
        total_len = len(new_data[self.target_iid_field])
        new_data[self.label_field] = np.zeros(total_len, dtype=np.int)
        new_data[self.label_field][:pos_len] = 1
        return new_data

    def get_pos_len_list(self):
        """
        Returns:
            np.ndarray or list: Number of positive item for each user in a training/evaluating epoch.
        """
        return np.ones(self.pr_end, dtype=np.int64)


class SequentialFullDataLoader(SequentialDataLoader):
    """:class:`SequentialFullDataLoader` is a sequential-dataloader with full sort. In order to speed up calculation,
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
        super().__init__(config, dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def _shuffle(self):
        self.logger.warnning('SequentialFullDataLoader can\'t shuffle')

    def _next_batch_data(self):
        interaction = super()._next_batch_data()
        tot_item_num = self.dataset.item_num
        inter_num = len(interaction)
        pos_idx = used_idx = interaction[self.target_iid_field] + torch.arange(inter_num) * tot_item_num
        pos_len_list = [1] * inter_num
        neg_len_list = [tot_item_num - 1] * inter_num
        return interaction, pos_idx, used_idx, pos_len_list, neg_len_list

    def get_pos_len_list(self):
        """
        Returns:
            np.ndarray or list: Number of positive item for each user in a training/evaluating epoch.
        """
        return np.ones(self.pr_end, dtype=np.int64)
