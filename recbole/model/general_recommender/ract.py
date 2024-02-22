# -*- coding: utf-8 -*-
# @Time   : 2021/2/16
# @Author : Haoran Cheng
# @Email  : chenghaoran29@foxmail.com

r"""
RaCT
################################################
Reference:
    Sam Lobel et al. "RaCT: Towards Amortized Ranking-Critical Training for Collaborative Filtering." in ICLR 2020.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class RaCT(GeneralRecommender, AutoEncoderMixin):
    r"""RaCT is a collaborative filtering model which uses methods based on actor-critic reinforcement learning for training.

    We implement the RaCT model with only user dataloader.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(RaCT, self).__init__(config, dataset)

        self.layers = config["mlp_hidden_size"]
        self.lat_dim = config["latent_dimension"]
        self.drop_out = config["dropout_prob"]
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]

        self.build_histroy_items(dataset)

        self.update = 0

        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][
            1:
        ]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)

        self.critic_layers = config["critic_layers"]
        self.metrics_k = config["metrics_k"]
        self.number_of_seen_items = 0
        self.number_of_unseen_items = 0
        self.critic_layer_dims = [3] + self.critic_layers + [1]

        self.input_matrix = None
        self.predict_matrix = None
        self.true_matrix = None
        self.critic_net = self.construct_critic_layers(self.critic_layer_dims)

        self.train_stage = config["train_stage"]
        self.pre_model_path = config["pre_model_path"]

        # parameters initialization
        assert self.train_stage in ["actor_pretrain", "critic_pretrain", "finetune"]
        if self.train_stage == "actor_pretrain":
            self.apply(xavier_normal_initialization)
            for p in self.critic_net.parameters():
                p.requires_grad = False
        elif self.train_stage == "critic_pretrain":
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            self.logger.info("Load pretrained model from", self.pre_model_path)
            self.load_state_dict(pretrained["state_dict"])
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            self.logger.info("Load pretrained model from", self.pre_model_path)
            self.load_state_dict(pretrained["state_dict"])
            for p in self.critic_net.parameters():
                p.requires_grad = False

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
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, rating_matrix):
        t = F.normalize(rating_matrix)

        h = F.dropout(t, self.drop_out, training=self.training) * (1 - self.drop_out)
        self.input_matrix = h
        self.number_of_seen_items = (h != 0).sum(dim=1)  # network input

        mask = (h > 0) * (t > 0)
        self.true_matrix = t * ~mask
        self.number_of_unseen_items = (self.true_matrix != 0).sum(
            dim=1
        )  # remaining input

        h = self.encoder(h)

        mu = h[:, : int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2) :]

        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        self.predict_matrix = z
        return z, mu, logvar

    def calculate_actor_loss(self, interaction):
        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, mu, logvar = self.forward(rating_matrix)

        # KL loss
        kl_loss = (
            -0.5 * (torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) * anneal
        )

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1)

        return ce_loss + kl_loss

    def construct_critic_input(self, actor_loss):
        critic_inputs = []
        critic_inputs.append(self.number_of_seen_items)
        critic_inputs.append(self.number_of_unseen_items)
        critic_inputs.append(actor_loss)
        return torch.stack(critic_inputs, dim=1)

    def construct_critic_layers(self, layer_dims):
        mlp_modules = []
        mlp_modules.append(nn.BatchNorm1d(3))
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.ReLU())
            else:
                mlp_modules.append(nn.Sigmoid())
        return nn.Sequential(*mlp_modules)

    def calculate_ndcg(self, predict_matrix, true_matrix, input_matrix, k):
        users_num = predict_matrix.shape[0]
        predict_matrix[input_matrix.nonzero(as_tuple=True)] = -np.inf
        _, idx_sorted = torch.sort(predict_matrix, dim=1, descending=True)

        topk_result = true_matrix[
            np.arange(users_num)[:, np.newaxis], idx_sorted[:, :k]
        ]

        number_non_zero = ((true_matrix > 0) * 1).sum(dim=1)

        tp = 1.0 / torch.log2(torch.arange(2, k + 2).type(torch.FloatTensor)).to(
            topk_result.device
        )
        DCG = (topk_result * tp).sum(dim=1)
        IDCG = torch.Tensor([(tp[: min(n, k)]).sum() for n in number_non_zero]).to(
            topk_result.device
        )
        IDCG = torch.maximum(0.1 * torch.ones_like(IDCG).to(IDCG.device), IDCG)

        return DCG / IDCG

    def critic_forward(self, actor_loss):
        h = self.construct_critic_input(actor_loss)
        y = self.critic_net(h)
        y = torch.squeeze(y)
        return y

    def calculate_critic_loss(self, interaction):
        actor_loss = self.calculate_actor_loss(interaction)
        y = self.critic_forward(actor_loss)
        score = self.calculate_ndcg(
            self.predict_matrix, self.true_matrix, self.input_matrix, self.metrics_k
        )

        mse_loss = (y - score) ** 2
        return mse_loss

    def calculate_ac_loss(self, interaction):
        actor_loss = self.calculate_actor_loss(interaction)
        y = self.critic_forward(actor_loss)
        return -1 * y

    def calculate_loss(self, interaction):
        # actor_pretrain
        if self.train_stage == "actor_pretrain":
            return self.calculate_actor_loss(interaction).mean()
        # critic_pretrain
        elif self.train_stage == "critic_pretrain":
            return self.calculate_critic_loss(interaction).mean()
        # finetune
        else:
            return self.calculate_ac_loss(interaction).mean()

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
