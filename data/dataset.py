# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : dataset.py

import os
import copy
import pandas as pd
import numpy as np
from .dataloader import *

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.token = config['dataset']
        self.dataset_path = config['data_path']

        self.support_types = set(['token', 'token_seq', 'float', 'float_seq'])
        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}

        self.inter_feat = None
        self.user_feat = None
        self.item_feat = None

        self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.token, self.dataset_path)

        self._remap_ID_all()

    def _load_data(self, token, dataset_path):
        user_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'user'))
        if os.path.isfile(user_feat_path):
            user_feat = self._load_feat(user_feat_path, 'user')
        else:
            # TODO logging user feat not exist
            user_feat = None

        item_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'item'))
        if os.path.isfile(item_feat_path):
            item_feat = self._load_feat(item_feat_path, 'item')
        else:
            # TODO logging item feat not exist
            item_feat = None

        inter_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'inter'))
        if not os.path.isfile(inter_feat_path):
            raise ValueError('File {} not exist'.format(inter_feat_path))
        inter_feat = self._load_feat(inter_feat_path, 'inter')

        uid_field = self.config['USER_ID_FIELD']
        if uid_field not in self.field2source:
            raise ValueError('user id field [{}] not exist in [{}]'.format(uid_field, self.token))
        else:
            self.field2source[uid_field] = 'user_id'

        iid_field = self.config['ITEM_ID_FIELD']
        if iid_field not in self.field2source:
            raise ValueError('item id field [{}] not exist in [{}]'.format(iid_field, self.token))
        else:
            self.field2source[iid_field] = 'item_id'

        return inter_feat, user_feat, item_feat

    def _load_feat(self, filepath, source):
        with open(filepath, 'r', encoding='utf-8') as file:
            head = file.readline().strip().split(self.config['field_separator'])
            field_names = []
            for field_type in head:
                field, ftype = field_type.split(':')
                # TODO user_id & item_id bridge check
                # TODO user_id & item_id not be set in config
                # TODO inter __iter__ loading
                if ftype not in self.support_types:
                    raise ValueError('Type {} from field {} is not supported'.format(ftype, field))
                self.field2source[field] = source
                self.field2type[field] = ftype
                field_names.append(field)

            # TODO checking num of col
            lines = []
            for line in file:
                lines.append(line.strip().split(self.config['field_separator']))

            ret = {}
            cols = map(list, zip(*lines))
            for i, col in enumerate(cols):
                field = field_names[i]
                ftype = self.field2type[field]
                # TODO not relying on str
                if ftype == 'float':
                    col = list(map(float, col))
                elif ftype == 'token_seq':
                    col = [_.split(self.config['seq_separator']) for _ in col]
                elif ftype == 'float_seq':
                    col = [list(map(float, _.split(self.config['seq_separator']))) for _ in col]
                ret[field] = col

        df = pd.DataFrame(ret)
        return df

    # TODO
    def filter_users(self):
        pass

    def _filter_inters(self, val, cmp):
        if val is not None:
            for field in val:
                if field not in self.field2type:
                    raise ValueError('field [{}] not defined in dataset'.format(field))
                self.inter_feat = self.inter_feat[cmp(self.inter_feat[field], val[field])]

        self.inter_feat = self.inter_feat.reset_index(drop=True)

    # TODO
    def filter_inters(self, lowest_val=None, highest_val=None, equal_val=None, not_equal_val=None):
        self._filter_inters(lowest_val, lambda x, y: x >= y)
        self._filter_inters(highest_val, lambda x, y: x <= y)
        self._filter_inters(equal_val, lambda x, y: x == y)
        self._filter_inters(not_equal_val, lambda x, y: x != y)

    def _remap_ID_all(self):
        for field in self.field2type:
            ftype = self.field2type[field]
            fsource = self.field2source[field]
            if ftype == 'token':
                self._remap_ID(fsource, field)
            elif ftype == 'token_seq':
                self._remap_ID_seq(fsource, field)

    def _remap_ID(self, source, field):
        feat_name = '{}_feat'.format(source.split('_')[0])
        feat = getattr(self, feat_name, pd.DataFrame(columns=[field]))
        if source in ['user_id', 'item_id']:
            df = pd.concat([self.inter_feat[field], feat[field]])
            new_ids, mp = pd.factorize(df)
            split_point = [len(self.inter_feat[field])]
            self.inter_feat[field], feat[field] = np.split(new_ids, split_point)
            self.field2id_token[field] = mp
        elif source in ['inter', 'user', 'item']:
            new_ids, mp = pd.factorize(feat[field])
            feat[field] = new_ids
            self.field2id_token[field] = mp

    def _remap_ID_seq(self, source, field):
        if source in ['inter', 'user', 'item']:
            feat_name = '{}_feat'.format(source)
            df = getattr(self, feat_name)
            split_point = np.cumsum(df[field].agg(len))[:-1]
            new_ids, mp = pd.factorize(df[field].agg(np.concatenate))
            new_ids = np.split(new_ids, split_point)
            df[field] = new_ids
            self.field2id_token[field] = mp

    def num(self, field):
        if field not in self.field2type:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        if self.field2type[field] != 'token' and self.field2type[field] != 'token_seq':
            raise ValueError('field [{}] is not a token type nor token_seq type'.format(field))
        return len(self.field2id_token[field])

    def __getitem__(self, index):
        df = self.inter_feat.loc[index]
        # TODO join user/item
        return df

    def __len__(self):
        return len(self.inter_feat)

    # def __iter__(self):
    #     return self

    # TODO next func
    # def next(self):
        # pass

    # TODO copy
    def copy(self, new_inter_feat):
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def split_by_ratio(self, ratio):
        tot_ratio = sum(ratio)
        ratio = [_ / tot_ratio for _ in ratio]

        tot_cnt = self.__len__()
        cnt = [int(ratio[i] * tot_cnt) for i in range(len(ratio))]
        cnt[-1] = tot_cnt - sum(cnt[0:-1])

        cur = 0
        next_ds = []
        for c in cnt:
            next_ds.append(self.copy(self.inter_feat[cur:cur+c]))
        return next_ds

    def shuffle(self):
        self.inter_feat = self.inter_feat.sample(frac=1).reset_index(drop=True)

    def sort(self, field_name):
        self.inter_feat = self.inter_feat.sort_values(by=field_name)

    # TODO
    def build(self, 
              inter_filter_lowest_val=None, inter_filter_highest_val=None, 
              split_by_ratio=None,
              train_batch_size=None, valid_batch_size=None, test_batch_size=None,
              pairwise=False,
              neg_sample_by=None, neg_sample_to=None
        ):
        # TODO
        self.filter_users()

        self.filter_inters(inter_filter_lowest_val, inter_filter_highest_val)

        assert len(split_by_ratio) == 3

        if split_by_ratio is not None:
            train_dataset, valid_dataset, test_dataset = self.split_by_ratio(split_by_ratio)

        # TODO
        train_loader = GeneralDataLoader(
            config=self.config,
            dataset=train_dataset,
            batch_size=train_batch_size,
            real_time_neg_sampling=False,
            shuffle=True,
            pairwise=pairwise,
            neg_sample_by=neg_sample_by,
        )

        test_loader = GeneralDataLoader(
            config=self.config,
            dataset=test_dataset,
            batch_size=test_batch_size,
            real_time_neg_sampling=False,
            neg_sample_to=neg_sample_to
        )

        valid_loader = GeneralDataLoader(
            config=self.config,
            dataset=valid_dataset,
            batch_size=valid_batch_size,
            real_time_neg_sampling=False,
            neg_sample_to=neg_sample_to
        )

        return train_loader, test_loader, valid_loader
