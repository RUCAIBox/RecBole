# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : dataset.py

import os
import pandas as pd
import numpy as np

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.token = config['data.name']
        self.dataset_path = config['data.path']

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

        # TODO
        self.n_users = len(self.token2id[self.config['data.USER_ID_FIELD']])
        self.n_items = len(self.token2id[self.config['data.ITEM_ID_FIELD']])

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

        return inter_feat, user_feat, item_feat

    def _load_feat(self, filepath, source):
        with open(filepath, 'r', encoding='utf-8') as file:
            head = file.readline().strip().split('\t')
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
                lines.append(line.strip().split('\t'))

            ret = {}
            cols = map(list, zip(*lines))
            for i, col in enumerate(cols):
                field = field_names[i]
                ftype = self.field2type[field]
                # TODO not relying on str
                if ftype == 'float':
                    col = list(map(float, col))
                elif ftype == 'token_seq':
                    col = [_.split(' ') for _ in col]
                elif ftype == 'float_seq':
                    col = [list(map(float, _.split(' '))) for _ in col]
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
        if source == 'inter':
            df = self.inter_feat
            new_ids, mp = pd.factorize(df[field])
            self.inter_feat[field] = new_ids
            self.token2id[field] = mp
        elif source == 'user':
            pass
        elif source == 'item':
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
