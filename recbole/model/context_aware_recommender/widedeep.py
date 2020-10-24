# -*- coding: utf-8 -*-
# @Time   : 2020/08/30
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn
# @File   : widedeep.py

r"""
WideDeep
#####################################################
Reference:
    Heng-Tze Cheng et al. "Wide & Deep Learning for Recommender Systems." in RecSys 2016.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.layers import MLPLayers
from recbole.model.abstract_recommender import ContextRecommender


class WideDeep(ContextRecommender):
    r"""WideDeep is a context-based recommendation model.
    It jointly trains wide linear models and deep neural networks to combine the benefits
    of memorization and generalization for recommender systems. The wide component is a generalized linear model
    of the form :math:`y = w^Tx + b`. The deep component is a feed-forward neural network. The wide component
    and deep component are combined using a weighted sum of their output log odds as the prediction,
    which is then fed to one common logistic loss function for joint training.
    """

    def __init__(self, config, dataset):
        super(WideDeep, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        # define layers and loss
        size_list = [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
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
        widedeep_all_embeddings = torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]
        batch_size = widedeep_all_embeddings.shape[0]
        fm_output = self.first_order_linear(interaction)

        deep_output = self.deep_predict_layer(
            self.mlp_layers(widedeep_all_embeddings.view(batch_size, -1)))
        output = self.sigmoid(fm_output + deep_output)
        return output.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
