# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

import random
import copy


class Sampler(object):
    def __init__(self, config, name, dataset):
        if not isinstance(name, list):
            name = [name]
        if not isinstance(dataset, list):
            dataset = [dataset]
        if len(name) != len(dataset):
            raise ValueError('name {} and dataset {} should have the same length'.format(name, dataset))

        self.config = config
        self.name = name
        self.dataset = dataset

        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']

        self.n_users = len(self.dataset[0].field2id_token[uid_field])
        self.n_items = len(self.dataset[0].field2id_token[iid_field])

        self.used_item_id = dict()
        last = [set() for i in range(self.n_users)]
        for name, dataset in zip(self.name, self.dataset):
            self.used_item_id[name] = copy.deepcopy(last)
            grouped_uid_iid = dataset.inter_feat.groupby(uid_field)[iid_field]
            for uid, iids in grouped_uid_iid:
                self.used_item_id[name][uid].update(iids.to_list())
            last = self.used_item_id[name]

    def sample_by_user_id(self, name, user_id, num=1):
        if name not in self.name:
            raise ValueError('name [{}] not exist'.format(name))
        if user_id not in range(self.n_users):
            raise ValueError('user_id [{}] not exist'.format(user_id))

        neg_item_id = []

        st = 0

        if num < 10:
            for i in range(num):
                cur = random.randint(st, self.n_items - 1)
                while cur in self.used_item_id[name][user_id]:
                    cur = random.randint(st, self.n_items - 1)
                neg_item_id.append(cur)
        else:
            tot = set(range(st, self.n_items)) - self.used_item_id[name][user_id]
            if len(tot) == num:
                neg_item_id = list(tot)
            else:
                neg_item_id = random.sample(tot, num)

        return neg_item_id


