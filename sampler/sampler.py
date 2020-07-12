# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

import random

class Sampler(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.used_item_id = {}
        self.n_users = len(self.dataset.field2id_token[self.config['USER_ID_FIELD']])
        self.n_items = len(self.dataset.field2id_token[self.config['ITEM_ID_FIELD']])

        for i in range(self.n_users):
            self.used_item_id[i] = set()

        uids = dataset.inter_feat[self.config['USER_ID_FIELD']].to_numpy()
        iids = dataset.inter_feat[self.config['ITEM_ID_FIELD']].to_numpy()
        assert len(uids) == len(iids)
        for i in range(len(uids)):
            uid = uids[i]
            iid = iids[i]
            self.used_item_id[uid].add(iid)

    def sample_by_user_id(self, user_id, num=1):
        if user_id not in self.used_item_id:
            raise ValueError('user_id [{}] not exist'.format(user_id))

        neg_item_id = []

        st = 0

        if num < 10:
            for i in range(num):
                cur = random.randint(st, self.n_items - 1)
                while cur in self.used_item_id[user_id]:
                    cur = random.randint(st, self.n_items - 1)
                neg_item_id.append(cur)
        else:
            tot = set(range(st, self.n_items)) - self.used_item_id[user_id]
            neg_item_id = random.sample(tot, num)

        return neg_item_id


