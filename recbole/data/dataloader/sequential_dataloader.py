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
from recbole.data.dataloader.neg_sample_mixin import NegSampleByMixin, NegSampleMixin
from recbole.data.interaction import Interaction, cat_interactions
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

    def __init__(self, config, dataset, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.time_field = dataset.time_field
        self.max_item_list_len = config['MAX_ITEM_LIST_LENGTH']

        list_suffix = config['LIST_SUFFIX']
        for field in dataset.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f'{field}_list_field', list_field)
                ftype = dataset.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, dataset.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                dataset.set_field_property(list_field, list_ftype, FeatureSource.INTERACTION, list_len)

        self.item_list_length_field = config['ITEM_LIST_LENGTH_FIELD']
        dataset.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)

        self.uid_list = dataset.uid_list
        self.item_list_index = dataset.item_list_index
        self.target_index = dataset.target_index
        self.item_list_length = dataset.item_list_length
        self.pre_processed_data = None

        super().__init__(config, dataset, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def data_preprocess(self):
        """Do data augmentation before training/evaluation.
        """
        self.pre_processed_data = self.augmentation(self.item_list_index, self.target_index, self.item_list_length)

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _shuffle(self):
        if self.real_time:
            new_index = torch.randperm(self.pr_end)
            self.uid_list = self.uid_list[new_index]
            self.item_list_index = self.item_list_index[new_index]
            self.target_index = self.target_index[new_index]
            self.item_list_length = self.item_list_length[new_index]
        else:
            self.pre_processed_data.shuffle()

    def _next_batch_data(self):
        cur_data = self._get_processed_data(slice(self.pr, self.pr + self.step))
        self.pr += self.step
        return cur_data

    def _get_processed_data(self, index):
        if self.real_time:
            cur_data = self.augmentation(
                self.item_list_index[index], self.target_index[index], self.item_list_length[index]
            )
        else:
            cur_data = self.pre_processed_data[index]
        return cur_data

    def augmentation(self, item_list_index, target_index, item_list_length):
        """Data augmentation.

        Args:
            item_list_index (numpy.ndarray): the index of history items list in interaction.
            target_index (numpy.ndarray): the index of items to be predicted in interaction.
            item_list_length (numpy.ndarray): history list length.

        Returns:
            dict: the augmented data.
        """
        new_length = len(item_list_index)
        new_data = self.dataset.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.dataset.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f'{field}_list_field')
                list_len = self.dataset.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.dataset.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.dataset.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        return new_data


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

    def __init__(
        self, config, dataset, sampler, neg_sample_args, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False
    ):
        super().__init__(
            config, dataset, sampler, neg_sample_args, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle
        )

    def _batch_size_adaptation(self):
        batch_num = max(self.batch_size // self.times, 1)
        new_batch_size = batch_num * self.times
        self.step = batch_num
        self.upgrade_batch_size(new_batch_size)

    def _next_batch_data(self):
        cur_data = self._get_processed_data(slice(self.pr, self.pr + self.step))
        cur_data = self._neg_sampling(cur_data)
        self.pr += self.step

        if self.user_inter_in_one_batch:
            cur_data_len = len(cur_data[self.uid_field])
            pos_len_list = np.ones(cur_data_len // self.times, dtype=np.int64)
            user_len_list = pos_len_list * self.times
            cur_data.set_additional_info(list(pos_len_list), list(user_len_list))
        return cur_data

    def _neg_sampling(self, data):
        if self.user_inter_in_one_batch:
            data_len = len(data[self.uid_field])
            data_list = []
            for i in range(data_len):
                uids = data[self.uid_field][i:i + 1]
                neg_iids = self.sampler.sample_by_user_ids(uids, self.neg_sample_by)
                cur_data = data[i:i + 1]
                data_list.append(self.sampling_func(cur_data, neg_iids))
            return cat_interactions(data_list)
        else:
            uids = data[self.uid_field]
            neg_iids = self.sampler.sample_by_user_ids(uids, self.neg_sample_by)
            return self.sampling_func(data, neg_iids)

    def _neg_sample_by_pair_wise_sampling(self, data, neg_iids):
        new_data = data.repeat(self.times)
        new_data.update(Interaction({self.neg_item_id: neg_iids}))
        return new_data

    def _neg_sample_by_point_wise_sampling(self, data, neg_iids):
        pos_inter_num = len(data)
        new_data = data.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_iids
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data

    def get_pos_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of positive item for each user in a training/evaluating epoch.
        """
        return np.ones(self.pr_end, dtype=np.int64)

    def get_user_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of all item for each user in a training/evaluating epoch.
        """
        return np.full(self.pr_end, self.times)


class SequentialFullDataLoader(NegSampleMixin, SequentialDataLoader):
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

    def __init__(
        self, config, dataset, sampler, neg_sample_args, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False
    ):
        super().__init__(
            config, dataset, sampler, neg_sample_args, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle
        )

    def _batch_size_adaptation(self):
        pass

    def _neg_sampling(self, inter_feat):
        pass

    def _shuffle(self):
        self.logger.warnning('SequentialFullDataLoader can\'t shuffle')

    def _next_batch_data(self):
        interaction = super()._next_batch_data()
        inter_num = len(interaction)
        pos_len_list = np.ones(inter_num, dtype=np.int64)
        user_len_list = np.full(inter_num, self.item_num)
        interaction.set_additional_info(pos_len_list, user_len_list)
        scores_row = torch.arange(inter_num).repeat(2)
        padding_idx = torch.zeros(inter_num, dtype=torch.int64)
        positive_idx = interaction[self.iid_field]
        scores_col_after = torch.cat((padding_idx, positive_idx))
        scores_col_before = torch.cat((positive_idx, padding_idx))
        return interaction, None, scores_row, scores_col_after, scores_col_before

    def get_pos_len_list(self):
        """
        Returns:
            numpy.ndarray or list: Number of positive item for each user in a training/evaluating epoch.
        """
        return np.ones(self.pr_end, dtype=np.int64)

    def get_user_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of all item for each user in a training/evaluating epoch.
        """
        return np.full(self.pr_end, self.item_num)
