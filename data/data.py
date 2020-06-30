# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : data.py

from torch.utils.data import DataLoader, Dataset

class Data(Dataset):
    def __init__(self, config, interaction):
        '''
        :param config(config.Config()): global configurations
        :param interaction(dict): dict of {
            Name: Tensor (batch, )
        }
        '''
        self.config = config
        self.interaction = interaction

        self._check()

        self.dataloader = DataLoader(
            dataset=self,
            batch_size=config['train.batch_size'],
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

    def split(self, ratio):
        '''
        :param ratio(float): A float in (0, 1), representing the first object's ratio
        :return: Two object of class Data, which has (ratio) and (1 - ratio), respectively
        '''
        div = int(ratio * self.__len__())
        first_inter = {}
        second_inter = {}
        for k in self.interaction:
            first_inter[k] = self.interaction[k][:div]
            second_inter[k] = self.interaction[k][div:]
        return Data(config=self.config, interaction=first_inter), \
               Data(config=self.config, interaction=second_inter)
