# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : data.py

import random
import torch
from torch.utils.data import DataLoader, Dataset
from sampler import Sampler

class Data(Dataset):
    def __init__(self, config, interaction, batch_size=1, sampler=None):
        '''
        :param config(config.Config()): global configurations
        :param interaction(dict): dict of {
            Name: Tensor (batch, )
        }
        '''
        self.config = config
        self.interaction = interaction
        self.batch_size = batch_size
        self.sampler = sampler

        self._check()

        self.dataloader = DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config['data.num_workers']
        )

    def _check(self):
        assert len(self.interaction.keys()) > 0
        for i, k in enumerate(self.interaction):
            if not i:
                self.length = len(self.interaction[k])
            else:
                assert len(self.interaction[k]) == self.length

    def __getitem__(self, index):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k][index]
        return ret

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.dataloader)

    def split_by_ratio(self, train_ratio, test_ratio, valid_ratio=0,
                       train_batch_size=None, test_batch_size=None, valid_batch_size=None
                      ):
        if train_ratio <= 0:
            raise ValueError('train ratio [{}] should be possitive'.format(train_ratio))
        if test_ratio <= 0:
            raise ValueError('test ratio [{}] should be possitive'.format(test_ratio))
        if valid_ratio < 0:
            raise ValueError('valid ratio [{}] should be none negative'.format(valid_ratio))

        tot_ratio = train_ratio + test_ratio + valid_ratio
        train_ratio /= tot_ratio
        test_ratio /= tot_ratio
        # valid_ratio /= tot_ratio

        train_cnt = int(train_ratio * self.__len__())
        if valid_ratio == 0:
            test_cnt = self.__len__() - train_cnt
            # valid_cnt = 0
        else:
            test_cnt = int(test_ratio * self.__len__())
            # valid_cnt = self.__len__() - train_cnt - test_cnt

        if train_batch_size is None:
            train_batch_size = self.batch_size
        if test_batch_size is None:
            test_batch_size = self.batch_size
        if valid_batch_size is None:
            valid_batch_size = self.batch_size

        train_inter = {}
        test_inter = {}
        valid_inter = {}
        for k in self.interaction:
            train_inter[k] = self.interaction[k][:train_cnt]
            test_inter[k] = self.interaction[k][train_cnt : train_cnt+test_cnt]
            if valid_ratio > 0:
                valid_inter[k] = self.interaction[k][train_cnt+test_cnt:]

        if valid_ratio > 0:
            return Data(config=self.config, interaction=train_inter, batch_size=train_batch_size, sampler=self.sampler), \
                   Data(config=self.config, interaction=test_inter, batch_size=test_batch_size, sampler=self.sampler), \
                   Data(config=self.config, interaction=valid_inter, batch_size=valid_batch_size, sampler=self.sampler)
        else:
            return Data(config=self.config, interaction=train_inter, batch_size=train_batch_size, sampler=self.sampler), \
                   Data(config=self.config, interaction=test_inter, batch_size=test_batch_size, sampler=self.sampler)

    def random_shuffle(self):
        idx = list(range(self.__len__()))
        random.shuffle(idx)
        next_inter = {}
        pass
        # TODO torch.xxx to random shuffle self.interaction

    def remove_lower_value_by_key(self, key, min_remain_value=0):
        new_inter = {}
        for k in self.interaction:
            new_inter[k] = []
        for i in range(self.__len__()):
            if self.interaction[key][i] >= min_remain_value:
                for k in self.interaction:
                    new_inter[k].append(self.interaction[k][i])
        for k in self.interaction:
            new_inter[k] = torch.stack(new_inter[k])

        new_sampler = Sampler(
            self.sampler.n_users, self.sampler.n_items,
            new_inter['user_id'], new_inter['item_id'],
            padding=self.sampler.padding, missing=self.sampler.missing
        )

        return Data(config=self.config, interaction=new_inter, batch_size=self.batch_size, sampler=new_sampler)

    def neg_sample_1by1(self):
        new_inter = {
            'user_id': [],
            'pos_item_id': [],
            'neg_item_id': []
        }
        for i in range(self.__len__()):
            uid = self.interaction['user_id'][i].item()
            new_inter['user_id'].append(uid)
            new_inter['pos_item_id'].append(self.interaction['item_id'][i].item())
            new_inter['neg_item_id'].append(self.sampler.sample_by_user_id(uid)[0])
        for k in new_inter:
            new_inter[k] = torch.LongTensor(new_inter[k])
        return Data(
            config=self.config,
            interaction=new_inter,
            batch_size=self.batch_size,
            sampler=self.sampler
        )

    # def neg_sample_to(self, num):
    #     new_inter = {
    #         'user_id': [],
    #         'item_list': [],
    #         'label': []
    #     }

    #     uid2itemlist = {}
    #     for i in range(self.__len__()):
    #         uid = self.interaction['user_id'][i].item()
    #         iid = self.interaction['item_id'][i].item()
    #         if uid not in uid2itemlist:
    #             uid2itemlist[uid] = []
    #         uid2itemlist[uid].append(iid)
    #     for uid in uid2itemlist:
    #         pos_num = len(uid2itemlist[uid])
    #         if pos_num >= num:
    #             uid2itemlist[uid] = uid2itemlist[uid][:num-1]
    #             pos_num = num - 1
    #         neg_item_id = self.sampler.sample_by_user_id(uid, num - pos_num)
    #         uid2itemlist[uid] += neg_item_id
    #         label = [1] * pos_num + [0] * (num - pos_num)
    #         new_inter['user_id'].append(uid)
    #         new_inter['item_list'].append(uid2itemlist[uid])
    #         new_inter['label'].append(label)
        
    #     for k in new_inter:
    #         new_inter[k] = torch.LongTensor(new_inter[k])

    #     return Data(config=self.config, interaction=new_inter, batch_size=self.batch_size, sampler=self.sampler)

    def neg_sample_to(self, num):
        new_inter = {
            'user_id': [],
            'item_id': [],
            'label': []
        }

        uid2itemlist = {}
        for i in range(self.__len__()):
            uid = self.interaction['user_id'][i].item()
            iid = self.interaction['item_id'][i].item()
            if uid not in uid2itemlist:
                uid2itemlist[uid] = []
            uid2itemlist[uid].append(iid)
        for uid in uid2itemlist:
            pos_num = len(uid2itemlist[uid])
            if pos_num >= num:
                uid2itemlist[uid] = uid2itemlist[uid][:num-1]
                pos_num = num - 1
            neg_item_id = self.sampler.sample_by_user_id(uid, num - pos_num)
            for iid in uid2itemlist[uid]:
                new_inter['user_id'].append(uid)
                new_inter['item_id'].append(iid)
                new_inter['label'].append(1)
            for iid in neg_item_id:
                new_inter['user_id'].append(uid)
                new_inter['item_id'].append(iid)
                new_inter['label'].append(0)
        
        for k in new_inter:
            new_inter[k] = torch.LongTensor(new_inter[k])

        return Data(config=self.config, interaction=new_inter, batch_size=self.batch_size, sampler=self.sampler)
