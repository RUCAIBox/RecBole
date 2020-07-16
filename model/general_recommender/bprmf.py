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
from model.layers import MLPLayers


class BPRMF(AbstractRecommender):

    def __init__(self, config, dataset):
        super(BPRMF, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = len(dataset.field2id_token[self.USER_ID])
        self.n_items = len(dataset.field2id_token[self.ITEM_ID])
        self.embedding_size = config['embedding_size']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        item_score = torch.mul(user_e, item_e).sum(dim=1)
        return item_score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        loss = - self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)
