# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 15:10
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : neumf.py

"""
Reference:
Xiangnan He et al., "Neural Collaborative Filtering." in WWW 2017.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from model.abstract_recommender import AbstractRecommender
from model.layers import MLPLayers


class NeuMF(AbstractRecommender):

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.LABEL = config['LABEL_FIELD']
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout = config['dropout']

        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers([2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout)
        self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], 1)
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

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)

        mf_output = torch.mul(user_mf_e, item_mf_e)     # (batch, embedding_size)
        mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))   # (batch, layers[-1])

        output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        return output.squeeze()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)
