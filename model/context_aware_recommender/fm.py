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

from model.abstract_recommender import ContextRecommender
from model.layers import BaseFactorizationMachine


class FM(ContextRecommender):

    def __init__(self, config, dataset):
        super(FM, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        x = []
        if sparse_embedding is not None:
            x.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            x.append(dense_embedding)
        x = torch.cat(x, dim=1)
        y = self.sigmoid(self.first_order_linear(interaction) + self.fm(self.embedding(x)))
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
