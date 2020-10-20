# -*- coding: utf-8 -*-
# @Time   : 2020/9/15 10:57
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# @File   : fnn.py

r"""
recbole.model.context_aware_recommender.fnn
################################################
Reference:
Weinan Zhang1 et al. "Deep Learning over Multi-field Categorical Data" in ECIR 2016
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from ..layers import MLPLayers
from .context_recommender import ContextRecommender


class FNN(ContextRecommender):
    """FNN which also called DNN is a basic version of CTR model that use mlp from field features to predict score.

    Note:
        Based on the experiments in the paper above, This implementation incorporate
        Dropout instead of L2 normalization to relieve over-fitting.
        Our implementation of FNN is a basic version without pretrain support.
        If you want to pretrain the feature embedding as the original paper,
        we suggest you to construct a advanced FNN model and train it in two-stage
        process with our FM model.
    """

    def __init__(self, config, dataset):
        super(FNN, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout = config['dropout']

        size_list = [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size

        self.mlp_layers = MLPLayers(size_list, self.dropout, activation='tanh', bn=False)  # use tanh as activation
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1, bias=True)

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
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        fnn_all_embeddings = torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]
        batch_size = fnn_all_embeddings.shape[0]

        output = self.predict_layer(self.mlp_layers(fnn_all_embeddings.view(batch_size, -1)))
        output = self.sigmoid(output)
        return output.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)

        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
