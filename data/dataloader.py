# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : dataloader.py

import pandas as pd
import torch
from sampler import Sampler
from .interaction import Interaction

class AbstractDataLoader(object):
    def __init__(self, config, dataset, batch_size):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = Sampler(config, dataset)

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError('Method [next] should be implemented.')

class PairwiseDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, batch_size, real_time_neg_sampling=True, neg_sample_to=None, neg_sample_by=None):
        super(PairwiseDataLoader, self).__init__(config, dataset, batch_size)

        self.real_time_neg_sampling = real_time_neg_sampling
        self.pr = 0

        self.neg_sample_to = neg_sample_to
        self.neg_sample_by = neg_sample_by
        if neg_sample_by is None and neg_sample_to is None:
            raise ValueError('neg_sample_to and neg_sample_by should not be None neither.')
        if neg_sample_by is not None and neg_sample_to is not None:
            raise ValueError('neg_sample_to and neg_sample_by cannot be given value the same time')

        if not real_time_neg_sampling:
            self._pre_neg_sampling()

    def __next__(self):
        if self.pr >= len(self.dataset):
            self.pr = 0
            raise StopIteration()
        cur_data = self.dataset[self.pr : self.pr+self.batch_size-1]
        self.pr += self.batch_size
        # TODO real time negative sampling
        if self.real_time_neg_sampling:
            pass
        cur_data = cur_data.to_dict(orient='list')
        for k in cur_data:
            ftype = self.dataset.field2type[k]
            if ftype.startswith('token'):
                cur_data[k] = torch.LongTensor(cur_data[k])
            elif ftype.startswith('float'):
                cur_data[k] = torch.FloatTensor(cur_data[k])
        return Interaction(cur_data)

    def _pre_neg_sampling(self):
        if self.neg_sample_by is not None:
            uids = self.dataset.inter_feat[self.config['USER_ID_FIELD']].to_list()
            # iids = self.dataset.inter_feat[self.config['ITEM_ID_FIELD']].to_list()
            if self.neg_sample_by == 1:
                neg_iids = []
                for uid in uids:
                    neg_iids.extend(self.sampler.sample_by_user_id(uid, self.neg_sample_by))
                neg_prefix = self.config['NEG_PREFIX']
                neg_item_id = neg_prefix + self.config['ITEM_ID_FIELD']
                self.dataset.inter_feat.insert(len(self.dataset.inter_feat.columns), neg_item_id, neg_iids)
                self.dataset.field2type[neg_item_id] = 'token'
                self.dataset.field2source[neg_item_id] = 'item_id'
                # TODO item_feat join
                if self.dataset.item_feat is not None:
                    pass
            else:
                raise NotImplementedError()
        # TODO
        elif self.neg_sample_to is not None:
            uid_field = self.config['USER_ID_FIELD']
            iid_field = self.config['ITEM_ID_FIELD']
            label_field = self.config['LABEL_FIELD']
            self.dataset.field2type[label_field] = 'float'
            self.dataset.field2source[label_field] = 'inter'
            new_inter = {
                uid_field: [],
                iid_field: [],
                label_field: []
            }

            uids = self.dataset.inter_feat[uid_field].to_list()
            iids = self.dataset.inter_feat[iid_field].to_list()
            uid2itemlist = {}
            for i in range(len(uids)):
                uid = uids[i]
                iid = iids[i]
                if uid not in uid2itemlist:
                    uid2itemlist[uid] = []
                uid2itemlist[uid].append(iid)
            for uid in uid2itemlist:
                pos_num = len(uid2itemlist[uid])
                if pos_num >= self.neg_sample_to:
                    uid2itemlist[uid] = uid2itemlist[uid][:self.neg_sample_to-1]
                    pos_num = self.neg_sample_to - 1
                neg_item_id = self.sampler.sample_by_user_id(uid, self.neg_sample_to - pos_num)
                for iid in uid2itemlist[uid]:
                    new_inter[uid_field].append(uid)
                    new_inter[iid_field].append(iid)
                    new_inter[label_field].append(1)
                for iid in neg_item_id:
                    new_inter[uid_field].append(uid)
                    new_inter[iid_field].append(iid)
                    new_inter[label_field].append(0)
            self.dataset.inter_feat = pd.DataFrame(new_inter)
