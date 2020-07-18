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
from torch.nn.init import xavier_normal_, constant_

from model.abstract_recommender import AbstractRecommender
from model.layers import FMEmbedding, FMFirstOrderLinear, BaseFactorizationMachine, MLPLayers


class NFM(AbstractRecommender):

    def __init__(self, config, dataset):
        super(NFM, self).__init__()

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

        size_list = [self.embedding_size] + self.mlp_hidden_size

        self.first_order_linear = FMFirstOrderLinear(self.field_dims, self.offsets)
        self.embedding = FMEmbedding(self.field_dims, self.offsets, self.embedding_size)
        self.fm = BaseFactorizationMachine(reduce_sum=False)
        self.mlp_layers = MLPLayers(size_list, self.dropout, activation='sigmoid')
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1, bias=False)
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
        emb_x = self.fm(self.embedding(x))

        y = self.predict_layer(self.mlp_layers(emb_x))+self.first_order_linear(x)
        y = self.sigmoid(y)
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
