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
import numpy as np

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization


class MultiDAE(GeneralRecommender):
    r"""MultiDAE is a item-based model collaborative filtering model that simultaneously rank all items for user .

    We implement the the MultiDAE model with only user dataloader.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MultiDAE, self).__init__(config, dataset)

        layers = config["mlp_hidden_size"]
        lat_dim = config['latent_dimendion']
        self.drop_out = config['dropout_prob']

        self.lat_dim = lat_dim
        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)

        self.encode_layer_dims = [self.n_items] + layers + [lat_dim]
        self.decode_layer_dims = [self.lat_dim] + self.encode_layer_dims[::-1][1:]

        self.encoder = self.mlp_layars(self.encode_layer_dims)
        self.decoder = self.mlp_layars(self.decode_layer_dims)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_rating_matrix(self, user_id):
        return torch.tensor(self.interaction_matrix[user_id.cpu(), :].toarray(), device=self.device)

    def mlp_layars(self, layer_dims):
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

        return scores[[user, item]]

    def full_sort_predict(self, interaction):

        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores = self.forward(rating_matrix)

        return scores.view(-1)
