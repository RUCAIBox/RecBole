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
    def __init__(self, config, dataset, batch_size):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = Sampler(config, dataset)
        self.pr = 0

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError('Method [next] should be implemented.')

    def set_batch_size(self, batch_size):
        if self.pr != 0:
            raise PermissionError('Cannot change dataloader\'s batch_size while iteration')
        self.batch_size = batch_size


class GeneralDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, batch_size, pairwise=False, shuffle=False, real_time_neg_sampling=True,
                 neg_sample_to=None, neg_sample_by=None):
        super(GeneralDataLoader, self).__init__(config, dataset, batch_size)

        self.pairwise = pairwise
        self.shuffle = shuffle
        self.real_time_neg_sampling = real_time_neg_sampling

        self.neg_sample_to = neg_sample_to
        self.neg_sample_by = neg_sample_by

        if neg_sample_by is not None and neg_sample_to is not None:
            raise ValueError('neg_sample_to and neg_sample_by cannot be given value the same time')

        if not real_time_neg_sampling:
            self._neg_sampling()

        if self.shuffle:
            self.dataset.shuffle()

    def __next__(self):
        if self.pr >= len(self.dataset):
            self.pr = 0
            raise StopIteration()
        cur_data = self.dataset[self.pr: self.pr + self.batch_size - 1]
        self.pr += self.batch_size
        if self.real_time_neg_sampling:
            if not self.pairwise:
                raise ValueError('real time neg sampling only support pairwise dataloader')
            uid_field = self.config['USER_ID_FIELD']
            iid_field = self.config['ITEM_ID_FIELD']
            cur_data = self._neg_sampling_by(uid_field, iid_field, cur_data)
        cur_data = cur_data.to_dict(orient='list')
        seqlen = self.dataset.field2seqlen
        for k in cur_data:
            ftype = self.dataset.field2type[k]
            if ftype == 'token':
                cur_data[k] = torch.LongTensor(cur_data[k])
            elif ftype == 'float':
                cur_data[k] = torch.FloatTensor(cur_data[k])
            elif ftype == 'token_seq':
                data = [torch.LongTensor(d[:seqlen[k]]) for d in cur_data[k]]  # TODO  cutting strategy?
                cur_data[k] = rnn_utils.pad_sequence(data, batch_first=True)
            elif ftype == 'float_seq':
                data = [torch.FloatTensor(d[:seqlen[k]]) for d in cur_data[k]]  # TODO  cutting strategy?
                cur_data[k] = rnn_utils.pad_sequence(data, batch_first=True)
            else:
                raise ValueError('Illegal ftype [{}]'.format(ftype))
        return Interaction(cur_data)

    def _neg_sampling(self):
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        if self.neg_sample_by is not None:
            sampling_func = self._neg_sampling_by
        elif self.neg_sample_to is not None:
            sampling_func = self._neg_sampling_to
        else:
            return
        self.dataset.inter_feat = sampling_func(uid_field, iid_field, self.dataset.inter_feat)

    def _neg_sampling_by(self, uid_field, iid_field, inter_feat):
        uids = inter_feat[uid_field].to_list()
        # iids = inter_feat[iid_field].to_list()
        # if self.neg_sample_by == 1:
        neg_iids = [self.sampler.sample_by_user_id(uid, self.neg_sample_by) for uid in uids]
        sampling_func = self._pair_wise_sampling if self.pairwise else self._point_wise_sampling
        return sampling_func(uid_field, iid_field, neg_iids, inter_feat)

    def _pair_wise_sampling(self, uid_field, iid_field, neg_iids, inter_feat):
        if self.neg_sample_by != 1:
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

    def _point_wise_sampling(self, uid_field, iid_field, neg_iids, inter_feat):
        neg_iids = list(np.array(neg_iids).T.ravel())
        neg_iids = inter_feat[iid_field].to_list() + neg_iids

        pos_inter_num = len(inter_feat)

        new_df = pd.concat([inter_feat] * (1 + self.neg_sample_by), ignore_index=True)
        new_df[iid_field] = neg_iids

        label_field = self.config['LABEL_FIELD']
        labels = pos_inter_num * [1] + self.neg_sample_by * pos_inter_num * [0]
        new_df[label_field] = labels

        return new_df

    # TODO
    def _neg_sampling_to(self, uid_field, iid_field, inter_feat):
        if self.neg_sample_to == -1:
            self.neg_sample_to = self.dataset.num(iid_field)
        if self.pairwise:
            raise ValueError('pairwise dataloader cannot neg sample to')
        user_num_in_one_batch = self.batch_size // self.neg_sample_to
        self.batch_size = (user_num_in_one_batch + 1) * self.neg_sample_to
        # TODO  batch size is changed

        label_field = self.config['LABEL_FIELD']
        self.dataset.field2type[label_field] = 'float'
        self.dataset.field2source[label_field] = 'inter'
        self.dataset.field2seqlen[label_field] = 1
        new_inter = {
            uid_field: [],
            iid_field: [],
            label_field: []
        }
        uid2itemlist = {}
        grouped_uid_iid = inter_feat.groupby(uid_field)[iid_field]
        for uid, iids in grouped_uid_iid:
            uid2itemlist[uid] = iids.to_list()
        for uid in uid2itemlist:
            pos_num = len(uid2itemlist[uid])
            if pos_num >= self.neg_sample_to:
                uid2itemlist[uid] = uid2itemlist[uid][:self.neg_sample_to - 1]
                pos_num = self.neg_sample_to - 1
            neg_num = self.neg_sample_to - pos_num
            neg_item_id = self.sampler.sample_by_user_id(uid, self.neg_sample_to - pos_num)

            new_inter[uid_field].extend([uid] * self.neg_sample_to)
            new_inter[iid_field].extend(uid2itemlist[uid] + neg_item_id)
            new_inter[label_field].extend([1] * pos_num + [0] * neg_num)

        return pd.DataFrame(new_inter)
