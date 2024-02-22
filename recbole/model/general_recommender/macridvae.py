# -*- coding: utf-8 -*-
# @Time   : 2020/12/23
# @Author : Yihong Guo
# @Email  : gyihong@hotmail.com

# UPDATE
# @Time    : 2021/6/30,
# @Author  : Xingyu Pan
# @email   : xy_pan@foxmail.com

r"""
MacridVAE
################################################
Reference:
    Jianxin Ma et al. "Learning Disentangled Representations for Recommendation." in NeurIPS 2019.

Reference code:
    https://jianxinma.github.io/disentangle-recsys.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class MacridVAE(GeneralRecommender, AutoEncoderMixin):
    r"""MacridVAE is an item-based collaborative filtering model that learns disentangled representations from user
    behavior and simultaneously ranks all items for each user.

    We implement the model following the original author.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MacridVAE, self).__init__(config, dataset)

        self.layers = config["encoder_hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.drop_out = config["dropout_prob"]
        self.kfac = config["kfac"]
        self.tau = config["tau"]
        self.nogb = config["nogb"]
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.regs = config["reg_weights"]
        self.std = config["std"]

        self.update = 0
        self.build_histroy_items(dataset)
        self.encode_layer_dims = (
            [self.n_items] + self.layers + [self.embedding_size * 2]
        )

        self.encoder = self.mlp_layers(self.encode_layer_dims)

        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.k_embedding = nn.Embedding(self.kfac, self.embedding_size)

        self.l2_loss = EmbLoss()
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, rating_matrix):
        cores = F.normalize(self.k_embedding.weight, dim=1)
        items = F.normalize(self.item_embedding.weight, dim=1)

        rating_matrix = F.normalize(rating_matrix)
        rating_matrix = F.dropout(rating_matrix, self.drop_out, training=self.training)

        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode

        probs = None
        mulist = []
        logvarlist = []
        for k in range(self.kfac):
            cates_k = cates[:, k].reshape(1, -1)
            # encoder
            x_k = rating_matrix * cates_k
            h = self.encoder(x_k)
            mu = h[:, : self.embedding_size]
            mu = F.normalize(mu, dim=1)
            logvar = h[:, self.embedding_size :]

            mulist.append(mu)
            logvarlist.append(logvar)

            z = self.reparameterize(mu, logvar)

            # decoder
            z_k = F.normalize(z, dim=1)
            logits_k = torch.matmul(z_k, items.transpose(0, 1)) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = probs_k if (probs is None) else (probs + probs_k)

        logits = torch.log(probs)

        return logits, mulist, logvarlist

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, mu, logvar = self.forward(rating_matrix)
        kl_loss = None
        for i in range(self.kfac):
            kl_ = -0.5 * torch.mean(torch.sum(1 + logvar[i] - logvar[i].exp(), dim=1))
            kl_loss = kl_ if (kl_loss is None) else (kl_loss + kl_)

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        if self.regs[0] != 0 or self.regs[1] != 0:
            return ce_loss + kl_loss * anneal + self.reg_loss()

        return ce_loss + kl_loss * anneal

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_1, reg_2 = self.regs[:2]
        loss_1 = reg_1 * self.item_embedding.weight.norm(2)
        loss_2 = reg_1 * self.k_embedding.weight.norm(2)
        loss_3 = 0
        for name, parm in self.encoder.named_parameters():
            if name.endswith("weight"):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        return scores.view(-1)
