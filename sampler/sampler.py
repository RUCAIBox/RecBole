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

        for i in range(self.dataset.n_users):
            self.used_item_id[i] = set()

        for interaction in self.dataset:
            uid = interaction[self.config['data.USER_ID_FIELD']]
            iid = interaction[self.config['data.ITEM_ID_FIELD']]
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


