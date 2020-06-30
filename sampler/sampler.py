# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

class Sampler(object):
    def __init__(self, n_users, n_items, user_id, item_id, padding=False, missing=False):
        self.used_item_id = {}
        self.n_users = n_users
        self.n_items = n_items
        self.padding = padding
        self.missing = missing

        if user_id.shape[0] != item_id.shape[0]:
            raise ValueError('user_id({}) and item_id({}) should have equal length.'.format(user_id.shape, item_id.shape))

        length = user_id.shape[0]

        for i in range(length):
            uid = user_id[i].item()
            iid = item_id[i].item()
            if uid not in self.used_item_id:
                self.used_item_id[uid] = set()
            self.used_item_id[uid].add(iid)

    def sample_by_user_id(self, user_id, num=1):
        pass

