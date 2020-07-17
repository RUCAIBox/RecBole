# -*- coding: utf-8 -*-
# @Time   : 2020/7/8 10:09
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : fm.py

"""
Reference:
Steffen Rendle et al., "Factorization Machines." in ICDM 2010.
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract_recommender import AbstractRecommender
from model.layers import FMEmbedding, FMFirstOrderLinear, BaseFactorizationMachine


class FM(AbstractRecommender):

    def __init__(self, config, dataset):
        super(FM, self).__init__()

        self.LABEL = config['LABEL_FIELD']
        self.embedding_size = config['embedding_size']
        self.field_names = list(dataset.field2id_token.keys())
        self.field_dims = [len(dataset.field2id_token[v]) for v in self.field_names]
        # todo: para: field2seqlen
        # self.field_seqlen = [dataset.field2seqlen[v] for v in self.field_names]
        self.field_seqlen = [1 for v in self.field_names]
        self.offsets = self._build_offsets()

        self.embedding = FMEmbedding(self.field_dims, self.offsets, self.embedding_size)
        self.first_order_linear = FMFirstOrderLinear(self.field_dims, self.offsets)
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def _build_offsets(self):
        offsets = []
        for i in range(len(self.field_names)):
            offsets += [self.field_dims[i]]
            offsets += [0] * (self.field_seqlen[i] - 1)
        offsets = np.array((0, *np.cumsum(offsets)[:-1]), dtype=np.long)
        return offsets

    def forward(self, interaction):
        x = []
        for field in self.field_names:
            # todo: check (batch) or (batch, 1)
            x.append(interaction[field].unsqueeze(1))
        x = torch.cat(x, dim=1)
        y = self.sigmoid(self.first_order_linear(x) + self.fm(self.embedding(x)))
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
