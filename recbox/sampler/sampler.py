# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

# UPDATE
# @Time   : 2020/8/17, 2020/8/31, 2020/8/31
# @Author : Xingyu Pan, Kaiyuan Li, Yupeng Hou
# @email  : panxy@ruc.edu.cn, tsotfsk@outlook.com, houyupeng@ruc.edu.cn

import random
import numpy as np


class Sampler(object):
    def __init__(self, config, phases, datasets, distribution='uniform'):
        legal_distribution = {'uniform', 'popularity'} 
        if distribution not in legal_distribution:
            raise ValueError('Distribution [{}] should in {}'.format(distribution, list(legal_distribution)))

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
        
        if distribution == 'uniform':
            self.random_item_list = list(range(self.n_items))
        elif distribution == 'popularity':
            self.random_item_list = []
            for dataset in datasets:
                self.random_item_list.extend(dataset.inter_feat[iid_field].values)
        else:
            raise NotImplementedError('Distribution [{}] has not been implemented'.format(distribution))

        random.shuffle(self.random_item_list)
        self.random_pr = 0
        self.random_item_list_length = len(self.random_item_list)

        self.full_set = set(range(self.n_items))
        self.used_item_id = dict()
        last = [set() for i in range(self.n_users)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for uid, iid in dataset.inter_feat[[uid_field, iid_field]].values:
                cur[uid].add(iid)
            last = self.used_item_id[phase] = cur

    def random_item(self):
        item = self.random_item_list[self.random_pr % self.random_item_list_length]
        self.random_pr += 1
        return item

    def sample_by_user_id(self, phase, user_id, num=1):
        try:
            neg_item_id = []
            used_item_id = self.used_item_id[phase][user_id]
            for step in range(self.random_item_list_length):
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
            for step in range(self.random_item_list_length):
                cur = self.random_item()
                if cur not in self.used_item_id[phase][user_id]:
                    return cur
        except KeyError:
            if phase not in self.phases:
                raise ValueError('phase [{}] not exist'.format(phase))
        except IndexError:
            if user_id < 0 or user_id >= self.n_users:
                raise ValueError('user_id [{}] not exist'.format(user_id))

    def sample_by_user_ids(self, phase, user_ids, num):
        try:
            user_num = len(user_ids)
            total_num = user_num * num
            neg_item_id = np.zeros(total_num, dtype=np.int64)
            used_item_id_list = np.repeat(self.used_item_id[phase][user_ids], num)
            for i, used_item_id in enumerate(used_item_id_list):
                cur = self.random_item()
                while cur in used_item_id:
                    cur = self.random_item()
                neg_item_id[i] = cur
            return neg_item_id
        except KeyError:
            if phase not in self.phases:
                raise ValueError('phase [{}] not exist'.format(phase))
        except IndexError:
            for user_id in user_ids:
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


class KGSampler(object):
    def __init__(self, config, phases, datasets, distribution='uniform'):
        legal_distribution = {'uniform', 'popularity'} 
        if distribution not in legal_distribution:
            raise ValueError('Distribution [{}] should in {}'.format(distribution, list(legal_distribution)))

        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError('phases {} and datasets {} should have the same length'.format(phases, datasets))

        self.config = config
        self.phases = phases
        self.datasets = datasets

        hid_field = self.config['HEAD_ENTITY_ID_FIELD']
        tid_field = self.config['TAIL_ENTITY_ID_FIELD']

        self.head_entities = set(self.datasets[0].head_entities)
        self.entities = self.datasets[0].entities_list
        self.entity_num = self.datasets[0].entity_num

        if distribution == 'uniform':
            self.random_entity_list = self.entities[:]
        elif distribution == 'popularity':
            self.random_entity_list = []
            for dataset in datasets:
                self.random_entity_list.extend(dataset.kg_feat[hid_field].values)
                self.random_entity_list.extend(dataset.kg_feat[tid_field].values)
        else:
            raise NotImplementedError('Distribution [{}] has not been implemented'.format(distribution))

        random.shuffle(self.random_entity_list)
        self.random_pr = 0
        self.random_entity_list_length = len(self.random_entity_list)

        self.full_set = set(self.entities)
        self.used_tail_eneity_id = dict()
        last = [set() for i in range(self.entity_num)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for hid, tid in dataset.kg_feat[[hid_field, tid_field]].values:
                cur[hid].add(tid)
            last = self.used_tail_eneity_id[phase] = cur

    def random_entity(self):
        entity = self.random_entity_list[self.random_pr % self.random_entity_list_length]
        self.random_pr += 1
        return entity

    def sample_by_entity_ids(self, phase, head_entity_ids, num=1):
        try:
            head_entity_num = len(head_entity_ids)
            total_num = head_entity_num * num
            neg_tail_entity_id = np.zeros(total_num, dtype=np.int64)
            used_tail_entity_id_list = np.repeat(self.used_tail_eneity_id[phase][head_entity_ids], num)
            for i, used_tail_entity_id in enumerate(used_tail_entity_id_list):
                cur = self.random_entity()
                while cur in used_tail_entity_id:
                    cur = self.random_entity()
                neg_tail_entity_id[i] = cur
            return neg_tail_entity_id
        except KeyError:
            if phase not in self.phases:
                raise ValueError('phase [{}] not exist'.format(phase))
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError('head_entity_id [{}] not exist'.format(head_entity_id))
