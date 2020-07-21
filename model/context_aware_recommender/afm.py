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
from torch.nn.init import xavier_normal_, constant_

from model.abstract_recommender import AbstractRecommender
from model.layers import FMEmbedding, FMFirstOrderLinear, AttLayer


class AFM(AbstractRecommender):

    def __init__(self, config, dataset):
        super(AFM, self).__init__()

        self.LABEL = config['LABEL_FIELD']
        self.embedding_size = config['embedding_size']
        self.attention_size = config['attention_size']
        self.dropout = config['dropout']
        self.field_names = list(dataset.field2id_token.keys())
        num_field = len(self.field_names)
        self.field_dims = [len(dataset.field2id_token[v]) for v in self.field_names]
        print(self.field_dims)
        # todo: para: field2seqlen
        # self.field_seqlen = [dataset.field2seqlen[v] for v in self.field_names]
        self.field_seqlen = [1 for v in self.field_names]
        self.offsets = self._build_offsets()
        self.num_pair = num_field * (num_field-1) / 2
        self.embedding = FMEmbedding(self.field_dims, self.offsets, self.embedding_size)
        self.first_order_linear = FMFirstOrderLinear(self.field_dims, self.offsets)
        self.attlayer = AttLayer(self.embedding_size, self.attention_size)
        self.p = nn.Parameter(torch.randn(self.embedding_size), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

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

    def build_cross(self, feat_emb):
        # num_pairs = num_fields * (num_fields-1) / 2
        row = []
        col = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]  # [batch_size, num_pairs, emb_dim]
        q = feat_emb[:, col]  # [batch_size, num_pairs, emb_dim]
        return p, q

    def afm_layer(self, infeature):
        """
        Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

        Output shape
        - 3D tensor with shape: ``(batch_size,1)`` .
        """
        p, q = self.build_cross(infeature)
        pair_wise_inter = torch.mul(p, q)  # [batch_size, num_pairs, emb_dim]

        att_signal = self.attlayer(pair_wise_inter).unsqueeze(dim=2)  # [batch_size, num_pairs, 1]

        att_inter = torch.mul(att_signal, pair_wise_inter)  # [batch_size, num_pairs, emb_dim]
        att_pooling = torch.sum(att_inter, dim=1)  # [batch_size, emb_dim]

        att_pooling = torch.mul(att_pooling, self.p)  # [batch_size, emb_dim]
        att_pooling = torch.sum(att_pooling, dim=1, keepdim=True)  # [batch_size, 1]

        return att_pooling

    def forward(self, interaction):
        x = []
        for field in self.field_names:
            # todo: check (batch) or (batch, 1)
            x.append(interaction[field].unsqueeze(1))
        x = torch.cat(x, dim=1)
        y = self.sigmoid(self.first_order_linear(x) + self.afm_layer(self.embedding(x)))
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
