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

import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers


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
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

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
        widedeep_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        batch_size = widedeep_all_embeddings.shape[0]
        fm_output = self.first_order_linear(interaction)

        deep_output = self.deep_predict_layer(
            self.mlp_layers(widedeep_all_embeddings.view(batch_size, -1))
        )
        output = fm_output + deep_output
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
