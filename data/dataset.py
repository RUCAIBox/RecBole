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
        self.dataset_path = config['dataset.path']

        self.support_types = set(['token', 'token_seq', 'float', 'float_seq'])
        self.field2type = {}
        self.field2source = {}
        self.token2id = {}

        self.inter_feat = None
        self.user_feat = None
        self.item_feat = None

        self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.token, self.dataset_path)

        self._filter_users()
        self._filter_inters()
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

        uid_field = self.config['data.USER_ID_FIELD']
        if uid_field not in self.field2source:
            raise ValueError('user id field [{}] not exist in [{}]'.format(uid_field, self.token))
        else:
            self.field2source[uid_field] = 'user_id'

        iid_field = self.config['data.ITEM_ID_FIELD']
        if iid_field not in self.field2source:
            raise ValueError('item id field [{}] not exist in [{}]'.format(iid_field, self.token))
        else:
            self.field2source[iid_field] = 'item_id'

        return inter_feat, user_feat, item_feat

    def _load_feat(self, filepath, source):
        with open(filepath, 'r', encoding='utf-8') as file:
            head = file.readline().strip().split(self.config['data.field_separator'])
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
                lines.append(line.strip().split(self.config['data.field_separator']))

            ret = {}
            cols = map(list, zip(*lines))
            for i, col in enumerate(cols):
                field = field_names[i]
                ftype = self.field2type[field]
                # TODO not relying on str
                if ftype == 'float':
                    col = list(map(float, col))
                elif ftype == 'token_seq':
                    col = [_.split(self.config['data.seq_separator']) for _ in col]
                elif ftype == 'float_seq':
                    col = [list(map(float, _.split(self.config['data.seq_separator']))) for _ in col]
                ret[field] = col

        df =  pd.DataFrame(ret)
        return df

    # TODO
    def _filter_users(self):
        pass

    # TODO
    def _filter_inters(self):
        if 'data.lowest_val' in self.config:
            for field in self.config['data.lowest_val']:
                if field not in self.field2type:
                    raise ValueError('field [{}] not defined in dataset'.format(field))
                self.inter_feat = self.inter_feat[self.inter_feat[field] >= self.config['data.lowest_val'][field]]
        if 'data.highest_val' in self.config:
            pass

        self.inter_feat = self.inter_feat.reset_index(drop=True)

    def _remap_ID_all(self):
        for field in self.field2type:
            ftype = self.field2type[field]
            fsource = self.field2source[field]
            if ftype == 'token':
                self._remap_ID(fsource, field)
            elif ftype == 'token_seq':
                self._remap_ID_seq(fsource, field)

    def _remap_ID(self, source, field):
        if source == 'inter' or source == 'user_id' or source == 'item_id':
            df = self.inter_feat
            new_ids, mp = pd.factorize(df[field])
            self.inter_feat[field] = new_ids
            self.token2id[field] = mp
        elif source == 'user' or source == 'user_id':
            pass
        elif source == 'item' or source == 'item_id':
            pass

    # TODO
    def _remap_ID_seq(self, source, field):
        pass

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

    # TODO
    def split(self):
        train_ratio = self.config['process.split_by_ratio.train_ratio']
        test_ratio = self.config['process.split_by_ratio.test_ratio']
        valid_ratio = self.config['process.split_by_ratio.valid_ratio']

        if train_ratio <= 0:
            raise ValueError('train ratio [{}] should be possitive'.format(train_ratio))
        if test_ratio <= 0:
            raise ValueError('test ratio [{}] should be possitive'.format(test_ratio))
        if valid_ratio < 0:
            raise ValueError('valid ratio [{}] should be none negative'.format(valid_ratio))

        tot_ratio = train_ratio + test_ratio + valid_ratio
        train_ratio /= tot_ratio
        test_ratio /= tot_ratio
        # valid_ratio /= tot_ratio

        train_cnt = int(train_ratio * self.__len__())
        if valid_ratio == 0:
            test_cnt = self.__len__() - train_cnt
            # valid_cnt = 0
        else:
            test_cnt = int(test_ratio * self.__len__())
            # valid_cnt = self.__len__() - train_cnt - test_cnt

        train_batch_size = self.config['model.train_batch_size']
        test_batch_size = self.config['model.test_batch_size']
        valid_batch_size = self.config['model.valid_batch_size']

        train_inter = self.inter_feat[:train_cnt]
        test_inter = self.inter_feat[train_cnt:train_cnt+test_cnt]
        valid_inter = self.inter_feat[train_cnt+test_cnt:]

        # TODO
        # NEED TO BE CHANGED A LOT
        # SHOULD BE DISCUSSED
        train_loader = PairwiseDataLoader(
            config=self.config,
            dataset=self.copy(train_inter),
            batch_size=train_batch_size,
            real_time_neg_sampling=False,
            neg_sample_by=1
        )

        test_loader = PairwiseDataLoader(
            config=self.config,
            dataset=self.copy(test_inter),
            batch_size=test_batch_size,
            real_time_neg_sampling=False,
            neg_sample_to=self.config['process.neg_sample_to.num']
        )

        valid_loader = PairwiseDataLoader(
            config=self.config,
            dataset=self.copy(valid_inter),
            batch_size=valid_batch_size,
            real_time_neg_sampling=False,
            neg_sample_to=self.config['process.neg_sample_to.num']
        )
        # NEED TO BE CHANGED A LOT
        # SHOULD BE DISCUSSED

        if valid_ratio is not None:
            return train_loader, test_loader, valid_loader
        else:
            return train_loader, test_loader
