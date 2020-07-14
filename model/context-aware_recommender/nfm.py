# -*- coding: utf-8 -*-
# @Time   : 2020/7/14 9:15
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# @File   : nfm.py

"""
Reference:
He X, Chua T S. "Neural factorization machines for sparse predictive analytics" in SIGIR 2017
"""

import torch
import torch.nn as nn
import numpy as np

from model.abstract_recommender import AbstractRecommender
from model.layers import FMEmbedding, FMFirstOrderLinear, BaseFactorizationMachine, MLPLayers


class NFM(AbstractRecommender):

    def __init__(self, config, dataset):
        super(NFM).__init__()

        self.embedding_size = config['embedding_size']
        self.layers = config['layers']
        self.dropout = config['dropout']
        self.field_names = list(dataset.token2id.keys())
        self.field_dims = [len(dataset.token2id[v]) for v in self.field_names]
        self.field_seqlen = [dataset.token2seqlen[v] for v in self.field_names]
        self.offsets = self._build_offsets()
        self.layers = [self.embedding_size] + self.layers

        self.first_order_linear = FMFirstOrderLinear(self.filed_dims, self.offsets)
        self.embedding = FMEmbedding(self.filed_dims, self.offsets, self.embedding_size)
        self.fm = BaseFactorizationMachine(reduce_sum=False)
        self.mlp_layers = MLPLayers(self.layers, self.dropout, activation='sigmoid')
        self.predict_layer = nn.Linear(self.layers[-1], 1, bias=False)

        self.loss = nn.BCELoss()

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
            x.append(interaction[field])
        x = torch.cat(x, dim=1)
        y = self.predict_layer(self.mlp_layers(self.embedding(x)))+self.first_order_linear(x)
        return y

    def train_model(self, interaction):
        label = interaction['LABEL']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
