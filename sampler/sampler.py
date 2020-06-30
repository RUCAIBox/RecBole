# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

import random

class Sampler(object):
    def __init__(self, n_users, n_items, user_id, item_id, padding=False, missing=False):
        self.used_item_id = {}
        self.n_users = n_users
        self.n_items = n_items
        self.padding = padding
        self.missing = missing

        if user_id.shape[0] != item_id.shape[0]:
            raise ValueError('user_id({}) and item_id({}) should have equal length.'.format(user_id.shape, item_id.shape))

        for i in range(self.n_users):
            self.used_item_id[i] = set()

        length = user_id.shape[0]

        for i in range(length):
            uid = user_id[i].item()
            iid = item_id[i].item()
            self.used_item_id[uid].add(iid)

    def sample_by_user_id(self, user_id, num=1):
        if user_id not in self.used_item_id:
            raise ValueError('user_id [{}] not exist'.format(user_id))

        neg_item_id = []

        if self.missing: st = 2
        elif self.padding: st = 1
        else: st = 0

        if num < 10:
            for i in range(num):
                cur = random.randint(st, self.n_items - 1)
                while cur in self.used_item_id[user_id]:
                    cur = random.randint(st, self.n_items - 1)
                neg_item_id.append(cur)
                self.used_item_id[user_id].add(cur)
        else:
            tot = set(range(st, self.n_items)) - self.used_item_id[user_id]
            neg_item_id = random.sample(tot, num)
        
        return neg_item_id


