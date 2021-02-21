# -*- coding: utf-8 -*-
# @Time   : 2021/2/28
# @Author : Lanling Xu
# @Email  : xulanling_sherry@163.com 

r"""
RecVAE
################################################
Reference:
    Shenbin, Ilya, et al. "RecVAE: A new variational autoencoder for Top-N recommendations with implicit feedback." Proceedings of the 13th International Conference on Web Search and Data Mining. 2020.

Reference code:
    https://github.com/ilya-shenbin/RecVAE
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class Swish(nn.Module):
    r"""Swish activation function:

    .. math::
        \text{Swish}(x) = \frac{x}{1 + \exp(-x)}
    """

    def forward(self, input):
        return input.mul(torch.sigmoid(input))


class RecVAE(GeneralRecommender):
    r"""Collaborative Denoising Auto-Encoder (RecVAE) is a recommendation model
    for top-N recommendation with implicit feedback.

    We implement the model following the original author
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(RecVAE, self).__init__(config, dataset)

        self.layers = config["encode_hidden_size"]
        self.latent_dim = config['latent_dimension']
        self.dropout_prob = config['dropout_prob']
        self.anneal_cap = config['anneal_cap']
        self.mixture_weights = config['mixture_weights']
        self.gamma = config['gamma']

        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

        self.encode_layer_dims = [self.n_items] + self.layers + [self.latent_dim]
        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = nn.Linear(self.latent_dim, self.n_items)

        # Composite prior
        self.mu_prior = nn.Parameter(torch.Tensor(1, self.latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        self.logvar_prior = nn.Parameter(torch.Tensor(1, self.latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, self.latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        self.encoder_old = self.mlp_layers(self.encode_layer_dims)
        self.encoder_old.requires_grad_(False)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).to(self.device) \
            .repeat_interleave(self.history_item_id.shape[1], dim=0)
        rating_matrix = torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        rating_matrix.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        return rating_matrix

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(Swish())
                mlp_modules.append(nn.LayerNorm(d_out, eps=0.1))
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def log_norm_pdf(self, x, mu, logvar):
        return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

    def forward(self, rating_matrix, drop_out=0):
        h = F.normalize(rating_matrix)
        h = F.dropout(h, drop_out, training=self.training)
        h = self.encoder(h)
        mu, logvar = h, h
        r = self.reparameterize(mu, logvar)
        z = self.decoder(r)
        return z, mu, logvar, r

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user)
        z, mu, logvar, r = self.forward(rating_matrix, self.dropout_prob)

        if self.gamma:
            norm = rating_matrix.sum(dim=-1)
            anneal = self.gamma * norm
        else:
            anneal = self.anneal_cap

        # Composite prior
        post_mu, post_logvar = self.encoder_old(rating_matrix), self.encoder_old(rating_matrix)
        stnd_prior = self.log_norm_pdf(r, self.mu_prior, self.logvar_prior)
        post_prior = self.log_norm_pdf(r, post_mu, post_logvar)
        unif_prior = self.log_norm_pdf(r, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        density_per_gaussian = torch.stack(gaussians, dim=-1)

        com_prior = torch.logsumexp(density_per_gaussian, dim=-1)

        # KL loss
        kl_loss = (self.log_norm_pdf(r, mu, logvar) - com_prior).sum(dim=-1).mul(anneal).mean()

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(dim=-1).mean()

        return kl_loss + ce_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _, _ = self.forward(rating_matrix, self.dropout_prob)

        return scores[[user, item]]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _, _ = self.forward(rating_matrix, self.dropout_prob)

        return scores.view(-1)
