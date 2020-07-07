# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : dataloader.py

from torch.utils import data
from sampler import Sampler

class AbstractDataLoader(data.DataLoader):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.sampler = Sampler(config, dataset)

class PairwiseDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset):
        super(PairwiseDataLoader, self).__init__(config, dataset)
