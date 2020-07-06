# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : dataset.py

from os.path import isdir, isfile
import torch
from .data import Data
from sampler import Sampler

class AbstractDataset(object):
    def __init__(self, config, padding=False, missing=False):
        self.config = config
        self.token = config['data.name']
        self.dataset_path = config['data.path']
        self.padding = padding
        self.missing = missing

        self.dataset, self.sampler, self.n_users, self.n_items = self._load_data(config)

    def __str__(self):
        return 'Dataset - {}'.format(self.token)

    def _load_data(self, config):
        '''
        :return: data.Data
        '''
        raise NotImplementedError('Func [_load_data] of [{}] has not been implemented'.format(
            self.__str__()
        ))

    def _download_dataset(self):
        '''
        Download dataset from url
        :return: path of the downloaded dataset
        '''
        pass

    def _get_n_users(self, user_id):
        self.user_ori2idx = {}
        self.user_idx2ori = []
        if self.missing:
            tot_users = 2
            self.user_ori2idx['padding'] = 0
            self.user_ori2idx['missing'] = 1
            self.user_idx2ori = ['padding', 'missing']
        elif self.padding:
            tot_users = 1
            self.user_ori2idx['padding'] = 0
            self.user_idx2ori = ['padding']
        else:
            tot_users = 0

        for uid in user_id:
            if uid not in self.user_ori2idx:
                self.user_ori2idx[uid] = tot_users
                self.user_idx2ori.append(uid)
                tot_users += 1
        return tot_users

    def _get_n_items(self, item_id):
        self.item_ori2idx = {}
        self.item_idx2ori = []
        if self.missing:
            tot_items = 2
            self.item_ori2idx['padding'] = 0
            self.item_ori2idx['missing'] = 1
            self.item_idx2ori = ['padding', 'missing']
        elif self.padding:
            tot_items = 1
            self.item_ori2idx['padding'] = 0
            self.item_idx2ori = ['padding']
        else:
            tot_items = 0

        for iid in item_id:
            if iid not in self.item_ori2idx:
                self.item_ori2idx[iid] = tot_items
                self.item_idx2ori.append(iid)
                tot_items += 1
        return tot_items
    
    def preprocessing(self, workflow=None):
        '''
        Preprocessing of the dataset
        '''
        cur = self.dataset
        train_data = test_data = valid_data = None
        for func in workflow['preprocessing']:
            if func == 'remove_lower_value_by_key':
                cur = cur.remove_lower_value_by_key(
                    key=self.config['process.remove_lower_value_by_key.key'],
                    min_remain_value=self.config['process.remove_lower_value_by_key.min_remain_value']
                )
            elif func == 'split_by_ratio':
                train_data, test_data, valid_data = cur.split_by_ratio(
                    train_ratio=self.config['process.split_by_ratio.train_ratio'],
                    test_ratio=self.config['process.split_by_ratio.test_ratio'],
                    valid_ratio=self.config['process.split_by_ratio.valid_ratio'],
                    train_batch_size=self.config['model.train_batch_size'],
                    test_batch_size=self.config['model.test_batch_size'],
                    valid_batch_size=self.config['model.valid_batch_size']
                )
                break

        for func in workflow['train']:
            if func == 'neg_sample_1by1':
                train_data = train_data.neg_sample_1by1()

        for func in workflow['test']:
            if func == 'neg_sample_to':
                test_data = test_data.neg_sample_to(num=self.config['process.neg_sample_to.num'])

        for func in workflow['valid']:
            if func == 'neg_sample_to':
                valid_data = valid_data.neg_sample_to(num=self.config['process.neg_sample_to.num'])
        
        return train_data, test_data, valid_data

class UIRTDataset(AbstractDataset):
    def __init__(self, config):
        super(UIRTDataset, self).__init__(config)

    def _load_data(self, config):
        if self.dataset_path:
            dataset_path = config['data.path']
        else:
            dataset_path = self._download_dataset(self.token)

        if not isfile(dataset_path):
            raise ValueError('[{}] is a illegal path.'.format(dataset_path))

        lines = []
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = map(int, line.strip().split('\t'))
                lines.append(line)
        user_id, item_id, rating, timestamp = map(list, zip(*lines))
        n_users = self._get_n_users(user_id)
        n_items = self._get_n_items(item_id)

        new_user_id = torch.LongTensor([self.user_ori2idx[_] for _ in user_id])
        new_item_id = torch.LongTensor([self.item_ori2idx[_] for _ in item_id])

        sampler = Sampler(n_users, n_items,
                          new_user_id, new_item_id, padding=self.padding, missing=self.missing)

        return Data(
            config=config,
            interaction={
                'user_id': new_user_id,
                'item_id': new_item_id,
                'rating': torch.LongTensor(rating),
                'timestamp': torch.LongTensor(timestamp)
            },
            sampler=sampler
        ), sampler, n_users, n_items

class ML100kDataset(UIRTDataset):
    def __init__(self, config):
        super(ML100kDataset, self).__init__(config)
