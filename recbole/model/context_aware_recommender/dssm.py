# -*- coding: utf-8 -*-
# @Time   : 2020/9/2
# @Author : Yingqian Min
# @Email  : gmqszyq@qq.com
# @File   : dssm.py


"""
recbole.model.context_aware_recommender.dssm
################################################
Reference:
PS Huang et al. "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data" in CIKM 2013.
"""


import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.layers import MLPLayers
from recbole.model.context_aware_recommender.context_recommender import ContextRecommender


class DSSM(ContextRecommender):

    def __init__(self, config, dataset):
        super(DSSM, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.user_feature_num = self.user_token_field_num + self.user_float_field_num + self.user_token_seq_field_num
        self.item_feature_num = self.item_token_field_num + self.item_float_field_num + self.item_token_seq_field_num
        user_size_list = [self.embedding_size * self.user_feature_num] + self.mlp_hidden_size
        item_size_list = [self.embedding_size * self.item_feature_num] + self.mlp_hidden_size

        self.user_mlp_layers = MLPLayers(user_size_list, self.dropout_prob, activation='tanh', bn=True)
        self.item_mlp_layers = MLPLayers(item_size_list, self.dropout_prob, activation='tanh', bn=True)

        self.loss = nn.BCELoss()
        self.sigmod = nn.Sigmoid()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        embed_result = self.double_tower_embed_input_fields(interaction)
        user_sparse_embedding, user_dense_embedding = embed_result[:2]
        item_sparse_embedding, item_dense_embedding = embed_result[2:]

        user = []
        if user_sparse_embedding is not None:
            user.append(user_sparse_embedding)
        if user_dense_embedding is not None and len(user_dense_embedding.shape) == 3:
            user.append(user_dense_embedding)

        embed_user = torch.cat(user, dim=1)

        item = []
        if item_sparse_embedding is not None:
            item.append(item_sparse_embedding)
        if item_dense_embedding is not None and len(item_dense_embedding.shape) == 3:
            item.append(item_dense_embedding)

        embed_item = torch.cat(item, dim=1)

        batch_size = embed_item.shape[0]
        user_dnn_out = self.user_mlp_layers(embed_user.view(batch_size, -1))
        item_dnn_out = self.item_mlp_layers(embed_item.view(batch_size, -1))
        score = torch.cosine_similarity(user_dnn_out, item_dnn_out, dim=1)

        sig_score = self.sigmod(score)
        return sig_score.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
