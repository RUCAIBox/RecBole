# -*- coding: utf-8 -*-
# @Time   : 2020/9/22 10:57
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
# @File   : pnn.py

"""
Reference:
Qu Y et al. "Product-based neural networks for user response prediction." in ICDM 2016

Note:

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from ..layers import MLPLayers
from .context_recommender import ContextRecommender


class PNN(ContextRecommender):

    def __init__(self, config, dataset):
        super(PNN, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout = config['dropout']
        self.use_inner = config['use_inner']
        self.use_outer = config['use_outer']
        self.d = self.mlp_hidden_size[0]
        self.product_linear = nn.Linear(self.embedding_size * self.num_feature_field, self.d, bias=False)
        self.product_bias = torch.nn.Parameter(torch.randn(self.d), requires_grad=True)
        if self.use_inner:
            self.product_inner = nn.Linear(self.embedding_size * self.num_feature_field, self.d, bias=False)
        if self.use_outer:
            self.product_outer = nn.Linear(self.embedding_size * self.embedding_size, self.d, bias=False)
        size_list = [self.d] + self.mlp_hidden_size

        self.mlp_layers = MLPLayers(size_list, self.dropout, bn=False)
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.relu = nn.ReLU()
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
        pnn_all_embeddings = torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]
        batch_size = pnn_all_embeddings.shape[0]
        # linear part
        lz = self.first_order_linear(pnn_all_embeddings.view(batch_size, -1))  # [batch_size,d]
        output = lz
        # second order part
        if self.use_inner:
            inner_product = self.product_inner(pnn_all_embeddings.view(batch_size, -1))  # [batch_size,d]
            output = output + inner_product
        if self.use_outer:
            sum_embedding = torch.sum(pnn_all_embeddings, dim=1)  # [batch_size, embed_dim]
            sum_embedding_matrix = torch.bmm(sum_embedding.unsqueeze(2), sum_embedding.unsqueeze(1))  # [batch_size,embed_dim,embed_dim]
            outer_product = self.product_outer(sum_embedding_matrix.view(batch_size, -1))  # [batch_size,d]
            output = output + outer_product
        output = output + self.product_bias  # [batch_size,d]
        output = self.relu(output)
        output = self.predict_layer(self.mlp_layers(output))  # [batch_size,1]
        output = self.sigmoid(output)
        return output.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)

        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
