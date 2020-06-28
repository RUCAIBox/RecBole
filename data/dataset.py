# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : dataset.py

from os.path import isdir, isfile
import torch
from .data import Data

class AbstractDataset(object):
    def __init__(self, config):
        self.token = config['data.name']
        self.dataset_path = config['data.path']
        self.dataset = self._load_data(config)

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
    
    def preprocessing(self, workflow=None):
        '''
        Preprocessing of the dataset
        :param workflow List(List(str, *args))
        '''
        cur = self.dataset
        for func, params in workflow:
            if func == 'split':
                cur = cur.split(*params)
        return cur

class ML100kDataset(AbstractDataset):
    def __init__(self, config):
        super(ML100kDataset, self).__init__(config)

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
        user_id, item_id, rating, timestamp = map(torch.LongTensor, zip(*lines))
        return Data(
            config=config,
            interaction={
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp
            }
        )
        
