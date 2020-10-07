# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

# UPDATE
# @Time   : 2020/8/17, 2020/8/31, 2020/10/6, 2020/9/18
# @Author : Xingyu Pan, Kaiyuan Li, Yupeng Hou, Yushuo Chen
# @email  : panxy@ruc.edu.cn, tsotfsk@outlook.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

import random
import copy
import numpy as np


class AbstractSampler(object):
    def __init__(self, distribution):
        self.distribution = distribution

        self.random_list = self.get_random_list()
        random.shuffle(self.random_list)
        self.random_pr = 0
        self.random_list_length = len(self.random_list)

        self.used_ids = self.get_used_ids()

    def get_random_list(self):
        raise NotImplementedError('method [get_random_list] should be implemented')

    def get_used_ids(self):
        raise NotImplementedError('method [get_used_ids] should be implemented')

    def random(self):
        item = self.random_list[self.random_pr % self.random_list_length]
        self.random_pr += 1
        return item

    def _sample_by_key_ids(self, key_ids, num, used_ids):
        key_num = len(key_ids)
        total_num = key_num * num
        neg_ids = np.zeros(total_num, dtype=np.int64)
        used_id_list = np.repeat(used_ids, num)
        for i, used_ids in enumerate(used_id_list):
            cur = self.random()
            while cur in used_ids:
                cur = self.random()
            neg_ids[i] = cur
        return neg_ids


class Sampler(AbstractSampler):
    def __init__(self, phases, datasets, distribution='uniform'):
        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError('phases {} and datasets {} should have the same length'.format(phases, datasets))

        self.phases = phases
        self.datasets = datasets

        self.uid_field = datasets[0].uid_field
        self.iid_field = datasets[0].iid_field

        self.n_users = datasets[0].user_num
        self.n_items = datasets[0].item_num

        super().__init__(distribution=distribution)

    def get_random_list(self):
        if self.distribution == 'uniform':
            return list(range(1, self.n_items))
        elif self.distribution == 'popularity':
            random_item_list = []
            for dataset in self.datasets:
                random_item_list.extend(dataset.inter_feat[self.iid_field].values)
            return random_item_list
        else:
            raise NotImplementedError('Distribution [{}] has not been implemented'.format(self.distribution))

    def get_used_ids(self):
        used_item_id = dict()
        last = [set() for i in range(self.n_users)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for uid, iid in dataset.inter_feat[[self.uid_field, self.iid_field]].values:
                cur[uid].add(iid)
            last = used_item_id[phase] = cur
        return used_item_id

    def set_phase(self, phase):
        if phase not in self.phases:
            raise ValueError('phase [{}] not exist'.format(phase))
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler

    def sample_by_user_ids(self, user_ids, num):
        try:
            return self._sample_by_key_ids(user_ids, num, self.used_ids[user_ids])
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.n_users:
                    raise ValueError('user_id [{}] not exist'.format(user_id))


class KGSampler(AbstractSampler):
    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset

        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field
        self.hid_list = dataset.head_entities
        self.tid_list = dataset.tail_entities

        self.head_entities = set(dataset.head_entities)
        self.entity_num = dataset.entity_num

        super().__init__(distribution=distribution)

    def get_random_list(self):
        if self.distribution == 'uniform':
            return list(range(1, self.entity_num))
        elif self.distribution == 'popularity':
            return list(self.hid_list) + list(self.tid_list)
        else:
            raise NotImplementedError('Distribution [{}] has not been implemented'.format(self.distribution))

    def get_used_ids(self):
        used_tail_entity_id = np.array([set() for i in range(self.entity_num)])
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)
        return used_tail_entity_id

    def sample_by_entity_ids(self, head_entity_ids, num=1):
        try:
            return self._sample_by_key_ids(head_entity_ids, num, self.used_ids[head_entity_ids])
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError('head_entity_id [{}] not exist'.format(head_entity_id))


class RepeatableSampler(AbstractSampler):
    def __init__(self, phases, dataset, distribution='uniform'):
        if not isinstance(phases, list):
            phases = [phases]
        self.phases = phases
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution)

    def get_random_list(self):
        if self.distribution == 'uniform':
            return list(range(1, self.item_num))
        elif self.distribution == 'popularity':
            return self.dataset.inter_feat[self.iid_field].values
        else:
            raise NotImplementedError('Distribution [{}] has not been implemented'.format(self.distribution))

    def get_used_ids(self):
        return np.array([set() for i in range(self.user_num)])

    def sample_by_user_ids(self, user_ids, num):
        try:
            return self._sample_by_key_ids(user_ids, num, self.used_ids[user_ids])
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.n_users:
                    raise ValueError('user_id [{}] not exist'.format(user_id))

    def set_phase(self, phase):
        if phase not in self.phases:
            raise ValueError('phase [{}] not exist'.format(phase))
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        return new_sampler
