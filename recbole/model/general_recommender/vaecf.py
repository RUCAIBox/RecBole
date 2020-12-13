# -*- coding: utf-8 -*-
# @Time   : 2020/12/8
# @Author : Yihong Guo
# @Email  : gyihong@hotmail.com

r"""
LINE
################################################
Reference:
    Jian Tang et al. "LINE: Large-scale Information Network Embedding." in WWW 2015.

Reference code:
    https://github.com/shenweichen/GraphEmbedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization




class VAECF(GeneralRecommender):
    r"""LINE is a graph embedding model.

    We implement the model with to train users and items embedding for recommendation.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(VAECF, self).__init__(config, dataset)

        layers = config["mlp_hidden_size"]
        lat_dim = config['latent_dimendion']
        self.drop_out = config['dropout_prob']
        self.kl_loss_weight = config['kl_loss_weight']
        self.is_vae = config['is_vae']

        self.lat_dim = lat_dim
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32).toarray()

        if self.is_vae:

            self.encode_layer_dims = [self.n_items] + layers + [lat_dim]
            self.decode_layer_dims = [int(self.lat_dim/2)]+self.encode_layer_dims[::-1][1:]

        else:

            self.encode_layer_dims = [self.n_items] + layers + [lat_dim]
            self.decode_layer_dims = [self.lat_dim] + self.encode_layer_dims[::-1][1:]

        self.encoder = self.mlp_layars(self.encode_layer_dims)
        self.decoder = self.mlp_layars(self.decode_layer_dims)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mlp_layars(self,layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=1)
            return mu + epsilon * std
        else:
            return mu

    def forward(self,rating_matrix):

        h = F.normalize(rating_matrix)

        h = F.dropout(h,self.drop_out,training=self.training)

        h = self.encoder(h)
        if self.is_vae:
            mu = h[:,:int(self.lat_dim/2)]
            logvar = h[:,int(self.lat_dim/2):]
            z = self.reparameterize(mu,logvar)
            z = self.decoder(z)
            return z, mu, logvar
        else:
            return self.decoder(h)

    def calculate_loss(self, interaction):

        user = interaction[self.USER_ID]

        rating_matrix = torch.tensor(self.interaction_matrix[user, :])

        if self.is_vae:
            z, mu, logvar = self.forward(rating_matrix)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        else:
            z = self.forward(rating_matrix)
            kl_loss = 0

        # CE
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        return ce_loss + kl_loss*self.kl_loss_weight

    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.interaction_matrix[user, :]
        if self.is_vae:
            scores, _, _ = self.forward(rating_matrix)
        else:
            scores = self.forward(rating_matrix)

        return scores[[user,item]]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = torch.tensor(self.interaction_matrix[user, :].reshape(-1,self.n_items))
        if self.is_vae:
            scores,_,_ = self.forward(rating_matrix)
        else:
            scores = self.forward(rating_matrix)

        return scores.view(-1)
