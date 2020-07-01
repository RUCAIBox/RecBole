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
from torch.nn.init import xavier_normal_

from model.abstract_recommender import AbstractRecommender
from model.layers import MLPLayers


class NeuMF(AbstractRecommender):

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__()

        self.embedding_size = config['model.embedding_size']
        self.layers = config['model.layers']
        self.dropout = config['model.dropout']
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

        self.user_mf_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.layers[0] // 2)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.layers[0] - self.layers[0] // 2)
        self.mlp_layers = MLPLayers(self.layers, self.dropout)
        self.predict_layer = nn.Linear(self.embedding_size + self.layers[-1], 1)
        self.loss = nn.BCEWithLogitsLoss()

        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.user_mf_embedding.weight)
        xavier_normal_(self.item_mf_embedding.weight)
        xavier_normal_(self.user_mlp_embedding.weight)
        xavier_normal_(self.item_mlp_embedding.weight)
        xavier_normal_(self.predict_layer.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)

        mf_output = torch.mul(user_mf_e, item_mf_e)
        mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))

        output = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        return output

    def train_model(self, user, item, label):
        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, user, item):
        return self.forward(user, item)
