# -*- coding: utf-8 -*-
# @Time   : 2020/7/8 10:33
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : deepfm.py

"""
Reference:
Huifeng Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." in IJCAI 2017.
"""

import torch
import torch.nn as nn
import numpy as np

from model.abstract_recommender import AbstractRecommender
from model.layers import FMEmbedding, FMFirstOrderLinear, BaseFactorizationMachine, MLPLayers


class DeepFM(AbstractRecommender):

    def __init__(self, config, dataset):
        super(DeepFM).__init__()

        self.embedding_size = config['model.embedding_size']
        self.layers = config['model.layers']
        self.dropout = config['model.dropout']
        self.field_names = list(dataset.token2id.keys())
        self.field_dims = [len(dataset.token2id[v]) for v in self.field_names]
        self.field_seqlen = [dataset.token2seqlen[v] for v in self.field_names]
        self.offsets = self._build_offsets()
        self.layers = [self.embedding_size * len(self.field_names)] + self.layers

        self.first_order_linear = FMFirstOrderLinear(self.filed_dims, self.offsets)
        self.embedding = FMEmbedding(self.filed_dims, self.offsets, self.embedding_size)
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        self.mlp_layers = MLPLayers(self.layers, self.dropout)
        self.deep_predict_layer = nn.Linear(self.layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
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
        embed_x = self.embedding(x)
        y_fm = self.first_order_linear(x) + self.fm(embed_x)
        # todo: how to deal with multi-hot feature (原论文明确规定每个field都是one-hot feature)
        y_deep = self.deep_predict_layer(
            self.mlp_layers(embed_x.view(-1, sum(self.field_seqlen) * self.embedding_size)))
        y = self.sigmoid(y_fm + y_deep)
        return y

    def train_model(self, interaction):
        label = interaction[LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
