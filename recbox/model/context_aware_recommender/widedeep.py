# -*- coding: utf-8 -*-
# @Time   : 2020/7/5 16:04
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : wide_and_deep.py

# UPDATE:
# @Time   : 2020/8/16,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

"""
Reference:
Cheng H T, Koc L, Harmsen J, et al., "Wide & deep learning for recommender systems." in RecSys 2016.
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from ..layers import MLPLayers
from .context_recommender import ContextRecommender


class WideDeep(ContextRecommender):

    def __init__(self, config, dataset):
        super(WideDeep, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout = config['dropout']

        size_list = [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size
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

    def forward(self, interaction):
        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        x = []
        if sparse_embedding is not None:
            x.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            x.append(dense_embedding)
        x = torch.cat(x, dim=1)  # [batch_size, num_field, embed_dim]
        batch_size = x.shape[0]
        y_fm = self.first_order_linear(interaction)

        y_deep = self.deep_predict_layer(
            self.mlp_layers(x.view(batch_size, -1)))
        y = self.sigmoid(y_fm + y_deep)
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
