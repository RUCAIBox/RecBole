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
from torch.nn.init import xavier_normal_, constant_

from model.abstract_recommender import ContextRecommender
from model.layers import FMEmbedding, FMFirstOrderLinear, BaseFactorizationMachine, MLPLayers


class DeepFM(ContextRecommender):

    def __init__(self, config, dataset):
        super(DeepFM, self).__init__()

        self.LABEL = config['LABEL_FIELD']
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout = config['dropout']
        self.field_names = list(dataset.field2id_token.keys())
        self.field_dims = [len(dataset.field2id_token[v]) for v in self.field_names]
        # todo: para: field2seqlen
        # self.field_seqlen = [dataset.field2seqlen[v] for v in self.field_names]
        self.field_seqlen = [1 for v in self.field_names]
        self.offsets = self._build_offsets()
        # todo: multi-hot len(field_names) or sum(field_seqlen)

        self.first_order_linear = FMFirstOrderLinear(self.field_dims, self.offsets)
        self.embedding = FMEmbedding(self.field_dims, self.offsets, self.embedding_size)
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [self.embedding_size * len(self.field_names)] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

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
        embed_x = self.embedding(x)
        y_fm = self.first_order_linear(x) + self.fm(embed_x)
        # todo: how to deal with multi-hot feature (原论文明确规定每个field都是one-hot feature)
        y_deep = self.deep_predict_layer(
            self.mlp_layers(embed_x.view(-1, sum(self.field_seqlen) * self.embedding_size)))
        y = self.sigmoid(y_fm + y_deep)
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
