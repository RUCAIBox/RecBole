# -*- coding: utf-8 -*-
# @Time   : 2020/12/14
# @Author : Yihong Guo
# @Email  : gyihong@hotmail.com

r"""
MultiDAE
################################################
Reference:
    Dawen Liang et al. "Variational Autoencoders for Collaborative Filtering." in WWW 2018.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.utils import InputType


class MultiDAE(GeneralRecommender, AutoEncoderMixin):
    r"""MultiDAE is an item-based collaborative filtering model that simultaneously ranks all items for each user.

    We implement the the MultiDAE model with only user dataloader.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MultiDAE, self).__init__(config, dataset)

        self.layers = config["mlp_hidden_size"]
        self.lat_dim = config["latent_dimension"]
        self.drop_out = config["dropout_prob"]

        self.build_histroy_items(dataset)

        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [self.lat_dim] + self.encode_layer_dims[::-1][1:]

        self.encoder = MLPLayers(self.encode_layer_dims, activation="tanh")
        self.decoder = self.mlp_layers(self.decode_layer_dims)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def forward(self, rating_matrix):
        h = F.normalize(rating_matrix)

        h = F.dropout(h, self.drop_out, training=self.training)

        h = self.encoder(h)
        return self.decoder(h)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        z = self.forward(rating_matrix)

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        return ce_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores = self.forward(rating_matrix)

        return scores.view(-1)
