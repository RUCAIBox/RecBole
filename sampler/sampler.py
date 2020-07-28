# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

import random
import copy


class Sampler(object):
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

        self.used_item_id = dict()
        last = [set() for i in range(self.n_users)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = copy.deepcopy(last)
            for row in dataset.inter_feat.itertuples():
                cur[getattr(row, uid_field)].add(getattr(row, iid_field))
            last = self.used_item_id[phase] = cur

    def sample_by_user_id(self, phase, user_id, num=1):
        if phase not in self.phases:
            raise ValueError('phase [{}] not exist'.format(phase))
        if user_id not in range(self.n_users):
            raise ValueError('user_id [{}] not exist'.format(user_id))

        neg_item_id = []

        st = 0

        if num < 10:
            for i in range(num):
                cur = random.randint(st, self.n_items - 1)
                while cur in self.used_item_id[phase][user_id]:
                    cur = random.randint(st, self.n_items - 1)
                neg_item_id.append(cur)
        else:
            tot = set(range(st, self.n_items)) - self.used_item_id[phase][user_id]
            if len(tot) <= num:
                neg_item_id = list(tot)
            else:
                neg_item_id = random.sample(tot, num)

        return neg_item_id


