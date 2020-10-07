# -*- coding: utf-8 -*-
# @Time   : 2020/8/13
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# @File   : lr.py

import torch.nn as nn
from torch.nn.init import xavier_normal_
from .context_recommender import ContextRecommender


class LR(ContextRecommender):
    def __init__(self, config, dataset):
        super(LR, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        output = self.sigmoid(self.first_order_linear(interaction))
        return output.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
