# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

import random


class Sampler(object):
    @profile
    def __init__(self, config, phases, datasets):
        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError('phases {} and datasets {} should have the same length'.format(phases, datasets))

        self.config = config
        self.phases = phases
        self.datasets = datasets

        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']

        self.n_users = self.datasets[0].user_num
        self.n_items = self.datasets[0].item_num

        self.random_item_list = list(range(self.n_items))
        random.shuffle(self.random_item_list)
        self.random_pr = 0

        self.full_set = set(range(self.n_items))
        self.used_item_id = dict()
        last = [set() for i in range(self.n_users)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = [set(s) for s in last]
            for uid, iid in dataset.inter_feat[[uid_field, iid_field]].values:
                cur[uid].add(iid)
            last = self.used_item_id[phase] = cur

    def random_item(self):
        item = self.random_item_list[self.random_pr % self.n_items]
        self.random_pr += 1
        return item

    def sample_by_user_id(self, phase, user_id, num=1):
        try:
            neg_item_id = []
            used_item_id = self.used_item_id[phase][user_id]
            for step in range(self.n_items):
                cur = self.random_item()
                if cur not in used_item_id:
                    neg_item_id.append(cur)
                    if len(neg_item_id) == num:
                        return neg_item_id
            return neg_item_id
        except KeyError:
            if phase not in self.phases:
                raise ValueError('phase [{}] not exist'.format(phase))
        except IndexError:
            if user_id < 0 or user_id >= self.n_users:
                raise ValueError('user_id [{}] not exist'.format(user_id))

    def sample_one_by_user_id(self, phase, user_id):
        try:
            for step in range(self.n_items):
                cur = self.random_item()
                if cur not in self.used_item_id[phase][user_id]:
                    return cur
        except KeyError:
            if phase not in self.phases:
                raise ValueError('phase [{}] not exist'.format(phase))
        except IndexError:
            if user_id < 0 or user_id >= self.n_users:
                raise ValueError('user_id [{}] not exist'.format(user_id))

    def sample_full_by_user_id(self, phase, user_id):
        try:
            return list(self.full_set - self.used_item_id[phase][user_id])
        except KeyError:
            if phase not in self.phases:
                raise ValueError('phase [{}] not exist'.format(phase))
        except IndexError:
            if user_id < 0 or user_id >= self.n_users:
                raise ValueError('user_id [{}] not exist'.format(user_id))
