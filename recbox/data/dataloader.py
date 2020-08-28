# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/8/27, 2020/8/27
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm

from ..utils import (
    DataLoaderType, EvaluatorType, FeatureSource, FeatureType, InputType,
    KGDataLoaderState)
from .interaction import Interaction


class AbstractDataLoader(object):
    def __init__(self, config, dataset,
                 batch_size=1, shuffle=False):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pr = 0
        self.dl_type = None

        self.join = self.dataset.join
        self.inter_matrix = self.dataset.inter_matrix
        self.num = self.dataset.num
        self.fields = self.dataset.fields
        self.field2type = self.dataset.field2type
        if self.dataset.uid_field:
            self.user_num = self.dataset.user_num
        if self.dataset.iid_field:
            self.item_num = self.dataset.item_num

    def __len__(self):
        raise NotImplementedError('Method [len] should be implemented')

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

    def _dataframe_to_interaction(self, data, *args):
        data = data.to_dict(orient='list')
        return self._dict_to_interaction(data, *args)

    def _dict_to_interaction(self, data, *args):
        seqlen = self.dataset.field2seqlen
        for k in data:
            ftype = self.dataset.field2type[k]
            if ftype == FeatureType.TOKEN:
                data[k] = torch.LongTensor(data[k])
            elif ftype == FeatureType.FLOAT:
                data[k] = torch.FloatTensor(data[k])
            elif ftype == FeatureType.TOKEN_SEQ:
                seq_data = [torch.LongTensor(d[:seqlen[k]]) for d in data[k]]
                data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            elif ftype == FeatureType.FLOAT_SEQ:
                seq_data = [torch.FloatTensor(d[:seqlen[k]]) for d in data[k]]
                data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            else:
                raise ValueError('Illegal ftype [{}]'.format(ftype))
        return Interaction(data, *args)

    def set_batch_size(self, batch_size):  # TODO batch size is useless...
        if self.pr != 0:
            raise PermissionError('Cannot change dataloader\'s batch_size while iteration')
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            # TODO  batch size is changed

    def get_item_feature(self):
        item_df = self.dataset.get_item_feature()
        return self._dataframe_to_interaction(item_df)


class GeneralDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.dl_type = DataLoaderType.ORIGIN
        self.step = batch_size

        self.dl_format = dl_format

        super(GeneralDataLoader, self).__init__(config, dataset, batch_size, shuffle)

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        return self._dataframe_to_interaction(cur_data)


class NegSampleBasedDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, sampler, phase, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        if neg_sample_args['strategy'] not in ['by', 'full']:
            raise ValueError('neg_sample strategy [{}] has not been implemented'.format(neg_sample_args['strategy']))

        super(NegSampleBasedDataLoader, self).__init__(config, dataset, batch_size, shuffle)

        self.dl_type = DataLoaderType.NEGSAMPLE

        self.sampler = sampler
        self.phase = phase
        self.neg_sample_args = neg_sample_args
        self.dl_format = dl_format
        self.real_time_neg_sampling = self.neg_sample_args['real_time']

        self._batch_size_adaptation()
        if not self.real_time_neg_sampling:
            self._pre_neg_sampling()

    def _batch_size_adaptation(self):
        raise NotImplementedError('Method [batch_size_adaptation] should be implemented.')

    def _pre_neg_sampling(self):
        raise NotImplementedError('Method [pre_neg_sampling] should be implemented.')

    def _neg_sampling(self, inter_feat):
        raise NotImplementedError('Method [neg_sampling] should be implemented.')


class GeneralIndividualDataLoader(NegSampleBasedDataLoader):
    def __init__(self, config, dataset, sampler, phase, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        if neg_sample_args['strategy'] != 'by':
            raise ValueError('neg_sample strategy in GeneralInteractionBasedDataLoader() should be `by`')
        if dl_format == InputType.PAIRWISE and neg_sample_args['by'] != 1:
            raise ValueError('Pairwise dataloader can only neg sample by 1')

        self.neg_sample_by = neg_sample_args['by']

        if dl_format == InputType.POINTWISE:
            self.times = 1 + self.neg_sample_by

            self.label_field = config['LABEL_FIELD']
            dataset.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        elif dl_format == InputType.PAIRWISE:
            self.times = 1

            neg_prefix = config['NEG_PREFIX']
            iid_field = config['ITEM_ID_FIELD']

            columns = [iid_field] if dataset.item_feat is None else dataset.item_feat.columns
            for item_feat_col in columns:
                neg_item_feat_col = neg_prefix + item_feat_col
                dataset.copy_field_property(neg_item_feat_col, item_feat_col)

        super(GeneralIndividualDataLoader, self).__init__(config, dataset, sampler, phase, neg_sample_args,
                                                          batch_size, dl_format, shuffle)

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def _batch_size_adaptation(self):
        if self.dl_format == InputType.PAIRWISE:
            self.step = self.batch_size
            return
        batch_num = max(self.batch_size // self.times, 1)
        new_batch_size = batch_num * self.times
        self.step = batch_num if self.real_time_neg_sampling else new_batch_size
        self.set_batch_size(new_batch_size)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        if self.real_time_neg_sampling:
            cur_data = self._neg_sampling(cur_data)
        return self._dataframe_to_interaction(cur_data)

    def _pre_neg_sampling(self):
        self.dataset.inter_feat = self._neg_sampling(self.dataset.inter_feat)

    def _neg_sampling(self, inter_feat):
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        uids = inter_feat[uid_field].to_list()
        neg_iids = self.sampler.sample_by_user_ids(self.phase, uids, self.neg_sample_by)
        if self.dl_format == InputType.POINTWISE:
            sampling_func = self._neg_sample_by_point_wise_sampling
        elif self.dl_format == InputType.PAIRWISE:
            sampling_func = self._neg_sample_by_pair_wise_sampling
        else:
            raise ValueError('`neg sampling by` with dl_format [{}] not been implemented'.format(self.dl_format))
        return sampling_func(uid_field, iid_field, neg_iids, inter_feat)

    def _neg_sample_by_pair_wise_sampling(self, uid_field, iid_field, neg_iids, inter_feat):
        neg_prefix = self.config['NEG_PREFIX']
        neg_item_id = neg_prefix + iid_field
        inter_feat.insert(len(inter_feat.columns), neg_item_id, neg_iids)

        if self.dataset.item_feat is not None:
            neg_item_feat = self.dataset.item_feat.add_prefix(neg_prefix)
            inter_feat = pd.merge(inter_feat, neg_item_feat,
                                  on=neg_item_id, how='left', suffixes=('_inter', '_item'))

        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, uid_field, iid_field, neg_iids, inter_feat):
        pos_inter_num = len(inter_feat)

        new_df = pd.concat([inter_feat] * self.times, ignore_index=True)
        new_df[iid_field].values[pos_inter_num:] = neg_iids

        labels = np.zeros(pos_inter_num * self.times, dtype=np.int64)
        labels[: pos_inter_num] = 1
        new_df[self.label_field] = labels

        return new_df


class GeneralGroupedDataLoader(GeneralIndividualDataLoader):
    def __init__(self, config, dataset, sampler, phase, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.uid2index, self.uid2items_num = dataset.uid2index

        super(GeneralGroupedDataLoader, self).__init__(config, dataset, sampler, phase, neg_sample_args,
                                                       batch_size, dl_format, shuffle)

    def _batch_size_adaptation(self):
        max_uid2inter_num = max(self.uid2items_num) * self.times
        batch_num = max(self.batch_size // max_uid2inter_num, 1)
        new_batch_size = batch_num * max_uid2inter_num
        self.step = batch_num
        self.set_batch_size(new_batch_size)

    @property
    def pr_end(self):
        return len(self.uid2index)

    def _shuffle(self):
        new_index = np.random.permutation(len(self.uid2index))
        self.uid2index = self.uid2index[new_index]
        self.uid2items_num = self.uid2items_num[new_index]

    def _next_batch_data(self):
        sampling_func = self._neg_sampling if self.real_time_neg_sampling else (lambda x: x)
        cur_data = []
        for uid, index in self.uid2index[self.pr: self.pr + self.step]:
            cur_data.append(sampling_func(self.dataset[index]))
        cur_data = pd.concat(cur_data, ignore_index=True)
        pos_len_list = self.uid2items_num[self.pr: self.pr + self.step]
        user_len_list = pos_len_list * self.times
        self.pr += self.step
        return self._dataframe_to_interaction(cur_data, list(pos_len_list), list(user_len_list))

    def _pre_neg_sampling(self):
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

    def get_pos_len_list(self):
        return self.uid2items_num


class GeneralFullDataLoader(NegSampleBasedDataLoader):
    def __init__(self, config, dataset, sampler, phase, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        if neg_sample_args['strategy'] != 'full':
            raise ValueError('neg_sample strategy in GeneralFullDataLoader() should be `full`')
        self.uid2index, self.uid2items_num = dataset.uid2index

        super().__init__(config, dataset, sampler, phase, neg_sample_args,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

        self.dl_type = DataLoaderType.FULL

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def _batch_size_adaptation(self):
        batch_num = max(self.batch_size // self.dataset.item_num, 1)
        new_batch_size = batch_num * self.dataset.item_num
        self.step = batch_num
        self.set_batch_size(new_batch_size)

    @property
    def pr_end(self):
        return len(self.uid2index)

    def _shuffle(self):
        raise NotImplementedError('GeneralFullDataLoader can\'t shuffle')

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

            used_item_id = self.sampler.used_item_id[self.phase][uid]
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

    def _pre_neg_sampling(self):
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

    def _next_batch_data(self):
        if not self.real_time_neg_sampling:
            slc = slice(self.pr, self.pr + self.step)
            idx = self.pr // self.step
            cur_data = self.user_tensor[slc], self.pos_idx[idx], self.used_idx[idx], \
                       self.pos_len_list[slc], self.neg_len_list[slc]
        else:
            cur_data = self._neg_sampling(self.uid2index[self.pr: self.pr + self.step])
        self.pr += self.step
        return cur_data

    def get_pos_len_list(self):
        return self.uid2items_num


class ContextDataLoader(GeneralDataLoader):
    pass


class ContextIndividualDataLoader(GeneralIndividualDataLoader):
    pass


class ContextGroupedDataLoader(GeneralGroupedDataLoader):
    pass


class SequentialDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        if dl_format != InputType.POINTWISE:
            raise ValueError('dl_format in Sequential DataLoader should be POINTWISE')

        self.dl_type = DataLoaderType.ORIGIN
        self.dl_format = dl_format
        self.step = batch_size
        self.real_time = config['real_time_process']

        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.time_field = dataset.time_field
        self.max_item_list_len = config['MAX_ITEM_LIST_LENGTH']
        self.stop_token_id = dataset.item_num - 1

        target_prefix = config['TARGET_PREFIX']
        list_suffix = config['LIST_SUFFIX']
        self.item_list_field = self.iid_field + list_suffix
        self.time_list_field = self.time_field + list_suffix
        self.position_field = config['POSITION_FIELD']
        self.target_iid_field = target_prefix + self.iid_field
        self.target_time_field = target_prefix + self.time_field
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

        if not self.real_time:
            self.pre_processed_data = self.augmentation(self.uid_list, self.item_list_field,
                                                        self.target_index, self.item_list_length)

        super(SequentialDataLoader, self).__init__(config, dataset, batch_size, shuffle)

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

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
        new_length = len(item_list_index)
        new_dict = {
            self.uid_field: uid_list,
            self.item_list_field: [],
            self.time_list_field: [],
            self.target_iid_field: self.dataset.inter_feat[self.iid_field][target_index].values,
            self.target_time_field: self.dataset.inter_feat[self.time_field][target_index].values,
            self.item_list_length_field: item_list_length,
        }
        if self.position_field:
            new_dict[self.position_field] = [np.arange(self.max_item_list_len)] * new_length
        for index in item_list_index:
            df = self.dataset.inter_feat[index]
            new_dict[self.item_list_field].append(np.append(df[self.iid_field].values, self.stop_token_id))
            new_dict[self.time_list_field].append(np.append(df[self.time_field].values, 0))
        return new_dict


class SequentialFullDataLoader(SequentialDataLoader):
    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        super(SequentialFullDataLoader, self).__init__(config, dataset, batch_size, dl_format, shuffle)

        self.dl_type = DataLoaderType.FULL

    def _shuffle(self):
        raise NotImplementedError('SequentialFullDataLoader can\'t shuffle')

    def _next_batch_data(self):
        interaction = super(SequentialFullDataLoader, self)._next_batch_data()
        tot_item_num = self.dataset.item_num
        inter_num = len(interaction)
        pos_idx = used_idx = interaction[self.target_iid_field] + torch.arange(inter_num) * tot_item_num
        pos_len_list = [1] * inter_num
        neg_len_list = [tot_item_num - 1] * inter_num
        return interaction, pos_idx, used_idx, pos_len_list, neg_len_list

    def get_pos_len_list(self):
        return np.ones(self.pr_end, dtype=np.int)


class KGDataLoader(NegSampleBasedDataLoader):

    def __init__(self, config, dataset, sampler, phase, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):

        super(KGDataLoader, self).__init__(config, dataset, sampler, phase, neg_sample_args,
                 batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)
        if neg_sample_args['strategy'] != 'by':
            raise ValueError('neg_sample strategy in KnowledgeBasedDataLoader() should be `by`')
        if dl_format != InputType.PAIRWISE and neg_sample_args['by'] != 1:
            raise ValueError('kg based dataloader must be pairwise and can only neg sample by 1')
        if shuffle is False:
            raise ValueError('kg based dataloader must shuffle the data')

        self.neg_sample_by = neg_sample_args['by']

        self.times = 1

        neg_prefix = config['NEG_PREFIX']
        iid_field = config['ITEM_ID_FIELD']
        tid_field = config['TAIL_ENTITY_ID_FIELD']

        # rec negative cols
        columns = [iid_field] if dataset.item_feat is None else dataset.item_feat.columns
        for item_feat_col in columns:
            neg_item_feat_col = neg_prefix + item_feat_col
            dataset.copy_field_property(neg_item_feat_col, item_feat_col)

        # kg negative cols
        neg_kg_col = neg_prefix + tid_field
        dataset.copy_field_property(neg_kg_col, tid_field)

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    @property
    def pr_end(self):
        # TODO 这个地方应该是取kg_data的len
        return len(self.dataset.kg_feat)

    def _shuffle(self):
        # TODO 这个地方应该是取kg_data的len
        self.dataset.kg_feat = self.dataset.kg_feat.sample(frac=1).reset_index(drop=True)

    def _next_batch_data(self):
        # TODO 这个地方应该取的kg_data
        cur_data = self.dataset.kg_feat[self.pr: self.pr + self.step]
        self.pr += self.step
        if self.real_time_neg_sampling:
            cur_data = self._neg_sampling(cur_data)
        return self._dataframe_to_interaction(cur_data)

    def _pre_neg_sampling(self):
        # TODO 这个地方应该是kg_data
        self.dataset.kg_feat = self._neg_sampling(self.dataset.kg_feat)

    def _neg_sampling(self, kg_feat):
        hid_field = self.config['HEAD_ENTITY_ID_FIELD']
        tid_field = self.config['TAIL_ENTITY_ID_FIELD']
        hids = kg_feat[hid_field].to_list()
        neg_tids = self.sampler.sample_by_entity_ids(self.phase, hids, self.neg_sample_by)
        return self._neg_sample_by_pair_wise_sampling(tid_field, neg_tids, kg_feat)

    def _neg_sample_by_pair_wise_sampling(self, tid_field, neg_tids, kg_feat):
        neg_prefix = self.config['NEG_PREFIX']
        neg_tail_entity_id = neg_prefix + tid_field
        kg_feat.insert(len(kg_feat.columns), neg_tail_entity_id, neg_tids)
        return kg_feat


class KnowledgeBasedDataLoader(AbstractDataLoader):

    def __init__(self, config, dataset, sampler, kg_sampler, phase, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):

        super(KnowledgeBasedDataLoader, self).__init__(config, dataset,
                                                       batch_size=batch_size, shuffle=shuffle)

        # using sampler
        self.general_dataloader = self.get_data_loader(config, dataset, sampler, phase, neg_sample_args,
                                                       batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

        # using kg_sampler and dl_format is pairwise
        self.kg_dataloader = KGDataLoader(config, dataset, kg_sampler, phase, neg_sample_args,
                                          batch_size=batch_size, dl_format=InputType.PAIRWISE, shuffle=False)

    def get_data_loader(self, **kwargs):
        phase = kwargs['phase']
        config = kwargs['config']
        if phase == 'train' or config['eval_type'] == EvaluatorType.INDIVIDUAL:
            return GeneralIndividualDataLoader(**kwargs)
        else:
            return GeneralGroupedDataLoader(**kwargs)

    @property
    def pr(self):
        return self.general_dataloader.pr

    @pr.setter
    def pr(self, value):
        self.general_dataloader.pr = value

    def __iter__(self):
        if not hasattr(self, 'state'):
            raise ValueError('The dataloader\'s state must be set when using the kg based dataloader')
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            # After the rec data ends, the kg data pointer needs to be cleared to zero
            self.kg_dataloader.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        if self.state in [KGDataLoaderState.RS, KGDataLoaderState.RSKG]:
            return self.general_dataloader.pr_end
        elif self.state == KGDataLoaderState.KG:
            return self.kg_dataloader.pr_end
        else:
            raise NotImplementedError('kg data loader has no state named [{}]'.format(self.state))

    def __len__(self):
        return len(self.general_dataloader)

    def _next_batch_data(self):
        if self.state == KGDataLoaderState.KG:
            return self.kg_dataloader._next_batch_data()
        elif self.state == KGDataLoaderState.RS:
            return self.general_dataloader._next_batch_data()
        elif self.state == KGDataLoaderState.RSKG:
            kg_data = self.kg_dataloader._next_batch_data()
            rec_data = self.general_dataloader._next_batch_data()
            return rec_data.update(kg_data)
        else:
            raise NotImplementedError('kg data loader has no state named [{}]'.format(self.state))

    def set_mode(self, state):
        self.state = state
