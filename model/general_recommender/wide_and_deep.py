# -*- coding: utf-8 -*-
# @Time   : 2020/7/5 16:04
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : wide_and_deep.py


import torch
import torch.nn as nn
import torch.functional as fn

from model.abstract_recommender import AbstractRecommender
from model.layers import MLPLayers

# wide_cols = ['education', 'age']
# wide_crossed_cols = [('education', 'age')]
# deep_embed_cols = ['education']
# deep_continuous_cols = ['age']


# todo: (wide optimizer: FTRL) (deep optimizer: AdaGrad)
class WideDeep(AbstractRecommender):

    def __init__(self, config, dataset):
        super(WideDeep, self).__init__()

        self.config = config
        self.embedding_size = config['model.embedding_size']
        self.layers = config['model.layers']
        self.dropout = config['model.dropout']
        self.wide_cols = config['wide_cols']
        self.wide_crossed_cols = config['crossed_cols']
        self.deep_embed_cols = config['deep_embed_cols']
        self.deep_continuous_cols = config['deep_continuous_cols']

        self.feature_class_num = dataset.feature_class_num
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

        self.category_embedding = {}
        for col in self.deep_embed_cols:
            self.category_embedding[col] = nn.Embedding(self.feature_class_num[col], self.embedding_size)
        self.mlp_layers = MLPLayers(self.layers, self.dropout)
        wide_output_dim = 0
        for col in self.wide_cols:
            wide_output_dim += self.feature_class_num[col]
        for col1, col2 in self.wide_crossed_cols:
            wide_output_dim += self.feature_class_num[col1] * self.feature_class_num[col2]
        self.linear = nn.Linear(self.layers[-1] + wide_output_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, interaction):
        # wide processor
        direct_vectors, crossed_vectors = [], []
        for col in self.wide_cols:
            batch_size = interaction[col].shape[0]
            feature = torch.unsqueeze(interaction[col], 1)
            one_hot_vector = torch.zeros(batch_size, self.feature_class_num[col]).scatter_(1, feature, 1)
            direct_vectors.append(one_hot_vector)
        # todo: cross feature transform to one-hot vector (n * m is redundance)
        for col1, col2 in self.wide_crossed_cols:
            batch_size = interaction[col1].shape[0]
            class_num1, class_num2 = self.feature_class_num[col1], self.feature_class_num[col2]
            feature = interaction[col1] * class_num2 + interaction[col2]
            one_hot_vector = torch.zeros(batch_size, class_num1 * class_num2).scatter_(1, feature, 1)
            crossed_vectors.append(one_hot_vector)
        wide_vectors = direct_vectors + crossed_vectors
        wide_vector = torch.cat(wide_vectors, 0)

        # deep processor
        embed_vectors, continuous_vectors = [], []
        for col in self.deep_embed_cols:
            embed_vector = self.category_embedding[col](interaction[col])
            embed_vectors.append(embed_vector)
        for col in self.deep_continuous_cols:
            continuous_vectors.append(interaction[col])
        deep_vectors = embed_vectors + continuous_vectors
        deep_vector = torch.cat(deep_vectors, 0)
        deep_vector = self.mlp_layers(deep_vector)

        output = self.linear(torch.cat([wide_vector, deep_vector], 1))
        return output

    def train_model(self, interaction):
        label = interaction[LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
