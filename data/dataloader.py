# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : dataloader.py

import operator
from functools import reduce
import pandas as pd
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from sampler import Sampler
from .interaction import Interaction


class AbstractDataLoader(object):
    def __init__(self, config, dataset,
                 batch_size=1, shuffle=False):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pr = 0

        if self.shuffle:
            self.dataset.shuffle()

    def __iter__(self):
        return self

    def __next__(self):
        if self.pr >= len(self.dataset):
            self.pr = 0
            raise StopIteration()
        cur_data = self._next_dataframe()
        return self._dataframe_to_interaction(cur_data)

    def _next_dataframe(self):
        raise NotImplementedError('Method [next_dataframe] should be implemented.')

    def _dataframe_to_interaction(self, data):
        data = data.to_dict(orient='list')
        seqlen = self.dataset.field2seqlen
        for k in data:
            ftype = self.dataset.field2type[k]
            if ftype == 'token':
                data[k] = torch.LongTensor(data[k])
            elif ftype == 'float':
                data[k] = torch.FloatTensor(data[k])
            elif ftype == 'token_seq':
                data = [torch.LongTensor(d[:seqlen[k]]) for d in data[k]]  # TODO  cutting strategy?
                data[k] = rnn_utils.pad_sequence(data, batch_first=True)
            elif ftype == 'float_seq':
                data = [torch.FloatTensor(d[:seqlen[k]]) for d in data[k]]  # TODO  cutting strategy?
                data[k] = rnn_utils.pad_sequence(data, batch_first=True)
            else:
                raise ValueError('Illegal ftype [{}]'.format(ftype))
        return Interaction(data)

    def set_batch_size(self, batch_size):
        if self.pr != 0:
            raise PermissionError('Cannot change dataloader\'s batch_size while iteration')
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            # TODO  batch size is changed


class NegSampleBasedDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, sampler, phase, neg_sample_args,
                 batch_size=1, dl_format='pointwise', shuffle=False):
        if dl_format not in ['pointwise', 'pairwise']:
            raise ValueError('dl_format [{}] has not been implemented'.format(dl_format))
        if neg_sample_args['strategy'] not in ['by', 'to']:
            raise ValueError('neg_sample strategy [{}] has not been implemented'.format(neg_sample_args['strategy']))

        super(NegSampleBasedDataLoader, self).__init__(config, dataset, batch_size, shuffle)

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


class InteractionBasedDataLoader(NegSampleBasedDataLoader):
    def __init__(self, config, dataset, sampler, phase, neg_sample_args,
                 batch_size=1, dl_format='pointwise', shuffle=False):
        if neg_sample_args['strategy'] != 'by':
            raise ValueError('neg_sample strategy in InteractionBasedDataLoader() should be `by`')
        super(InteractionBasedDataLoader, self).__init__(config, dataset, sampler, phase, neg_sample_args,
                                                         batch_size, dl_format, shuffle)

    def _batch_size_adaptation(self):
        if self.dl_format == 'pairwise':
            self.step = self.batch_size
            return
        batch_num = self.batch_size // self.neg_sample_args['by']
        new_batch_size = (batch_num + 1) * self.neg_sample_args['by']
        self.step = batch_num + 1 if self.real_time_neg_sampling else new_batch_size
        self.set_batch_size(new_batch_size)

    def _next_dataframe(self):
        cur_data = self.dataset[self.pr: self.pr + self.step - 1]
        self.pr += self.step
        if self.real_time_neg_sampling:
            cur_data = self._neg_sampling(cur_data)
        return cur_data

    def _pre_neg_sampling(self):
        self.dataset.inter_feat = self._neg_sampling(self.dataset.inter_feat)

    def _neg_sampling(self, inter_feat):
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        uids = inter_feat[uid_field].to_list()
        neg_iids = [self.sampler.sample_by_user_id(self.phase, uid, self.neg_sample_args['by']) for uid in uids]
        if self.dl_format == 'pointwise':
            sampling_func = self._neg_sample_by_point_wise_sampling
        elif self.dl_format == 'pairwise':
            sampling_func = self._neg_sample_by_pair_wise_sampling
        else:
            raise ValueError('`neg sampling by` with dl_format [{}] not been implemented'.format(self.dl_format))
        return sampling_func(uid_field, iid_field, neg_iids, inter_feat)

    def _neg_sample_by_pair_wise_sampling(self, uid_field, iid_field, neg_iids, inter_feat):
        if self.neg_sample_args['by'] != 1:
            raise ValueError('Pairwise dataloader can only neg sample by 1')
        neg_prefix = self.config['NEG_PREFIX']
        neg_item_id = neg_prefix + iid_field
        neg_iids = np.array(neg_iids).ravel()

        inter_feat.insert(len(inter_feat.columns), neg_item_id, neg_iids)
        self.dataset.field2type[neg_item_id] = 'token'
        self.dataset.field2source[neg_item_id] = 'item_id'
        self.dataset.field2seqlen[neg_item_id] = self.dataset.field2seqlen[iid_field]

        if self.dataset.item_feat is not None:
            neg_item_feat = self.dataset.item_feat.add_prefix(neg_prefix)
            inter_feat = pd.merge(inter_feat, neg_item_feat,
                                  on=neg_item_id, how='left', suffixes=('_inter', '_item'))
            for neg_item_feat_col, item_feat_col in zip(neg_item_feat.columns, self.dataset.item_feat.columns):
                self.dataset.field2type[neg_item_feat_col] = self.dataset.field2type[item_feat_col]
                self.dataset.field2source[neg_item_feat_col] = self.dataset.field2source[item_feat_col]
                self.dataset.field2seqlen[neg_item_feat_col] = self.dataset.field2seqlen[item_feat_col]

        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, uid_field, iid_field, neg_iids, inter_feat):
        neg_iids = list(np.array(neg_iids).T.ravel())
        neg_iids = inter_feat[iid_field].to_list() + neg_iids

        pos_inter_num = len(inter_feat)

        new_df = pd.concat([inter_feat] * (1 + self.neg_sample_args['by']), ignore_index=True)
        new_df[iid_field] = neg_iids

        label_field = self.config['LABEL_FIELD']
        labels = pos_inter_num * [1] + self.neg_sample_args['by'] * pos_inter_num * [0]
        new_df[label_field] = labels

        return self.dataset.join(new_df) if self.real_time_neg_sampling else new_df


class GroupedDataLoader(NegSampleBasedDataLoader):
    def __init__(self, config, dataset, sampler, phase, neg_sample_args,
                 batch_size=1, dl_format='pointwise', shuffle=False):
        if neg_sample_args['strategy'] != 'to':
            raise ValueError('neg_sample strategy in GroupedDataLoader() should be `to`')
        if dl_format == 'pairwise':
            raise ValueError('pairwise dataloader cannot neg sample to')

        self.uid2items = dataset.uid2items
        super(GroupedDataLoader, self).__init__(config, dataset, sampler, phase, neg_sample_args,
                                                batch_size, dl_format, shuffle)

    def _batch_size_adaptation(self):
        if self.neg_sample_args['to'] == -1:
            self.neg_sample_args['to'] = self.dataset.item_num
        batch_num = self.batch_size // self.neg_sample_args['to']
        new_batch_size = (batch_num + 1) * self.neg_sample_args['to']
        self.step = batch_num + 1 if self.real_time_neg_sampling else new_batch_size

    @property
    def next_pr(self):
        return self.next[self.pr]

    def _next_dataframe(self):
        if self.real_time_neg_sampling:
            cur_data = self._neg_sampling(self.uid2items[self.pr: self.pr + self.step - 1])
            self.pr += self.step
        else:
            cur_data = self.dataset[self.pr: self.next_pr - 1]
            self.pr = self.next_pr
        return cur_data

    def _pre_neg_sampling(self):
        self.dataset.inter_feat = self._neg_sampling(self.uid2items)

    def _neg_sampling(self, uid2items):
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        label_field = self.config['LABEL_FIELD']
        self.dataset.field2type[label_field] = 'float'
        self.dataset.field2source[label_field] = 'inter'
        self.dataset.field2seqlen[label_field] = 1
        new_inter = {
            uid_field: [],
            iid_field: [],
            label_field: []
        }

        for i, uid in enumerate(uid2items):
            pos_item_id = uid2items[uid][:self.neg_sample_args['to'] - 1]
            pos_num = len(pos_item_id)
            neg_item_id = self.sampler.sample_by_user_id(self.phase, uid, self.neg_sample_args['to'] - pos_num)
            neg_num = len(neg_item_id)

            new_inter[uid_field].extend([uid] * (pos_num + neg_num))
            new_inter[iid_field].extend(pos_item_id + neg_item_id)
            new_inter[label_field].extend([1] * pos_num + [0] * neg_num)

            if not self.real_time_neg_sampling and i % self.step == 0:
                if i == 0:
                    self.next = dict()
                    last_pr = 0
                else:
                    self.next[last_pr] = len(new_inter[uid_field])
                    last_pr = len(new_inter[uid_field])

        new_inter = pd.DataFrame(new_inter)
        if not self.real_time_neg_sampling:
            self.next[last_pr] = len(new_inter)
            return new_inter
        else:
            return self.dataset.join(new_inter)


def get_data_loader(neg_sample_args):
    if neg_sample_args['strategy'] == 'by':
        return InteractionBasedDataLoader
    elif neg_sample_args['strategy'] == 'to':
        return GroupedDataLoader
    else:
        raise ValueError('neg_sample strategy [{}] has not been implemented'.format(neg_sample_args['strategy']))
