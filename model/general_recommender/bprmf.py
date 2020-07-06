# -*- coding: utf-8 -*-
# @Time   : 2020/6/25 16:28
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : bprmf.py

"""
Reference:
Steffen Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract_recommender import AbstractRecommender
from model.loss import BPRLoss


class BPRMF(AbstractRecommender):

    def __init__(self, config, dataset):
        super(BPRMF, self).__init__()

        self.embedding_size = config['model.embedding_size']
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.user_embedding.weight)
        xavier_normal_(self.item_embedding.weight)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        item_score = torch.mul(user_e, item_e).sum(dim=1)
        return item_score

    def train_model(self, interaction):
        user = interaction[USER_ID]
        pos_item = interaction[ITEM_ID]
        neg_item = interaction[NEG_ITEM_ID]

        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        loss = - self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction[USER_ID]
        item = interaction[ITEM_ID]
        return self.forward(user, item)
