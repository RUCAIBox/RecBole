# -*- coding: utf-8 -*-
# @Time   : 2020/08/30
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn
# @File   : lr.py

r"""
LR
#####################################################
Reference:
    Matthew Richardson et al. "Predicting Clicks Estimating the Click-Through Rate for New Ads." in WWW 2007.
"""

import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import ContextRecommender


class LR(ContextRecommender):
    r"""LR is a context-based recommendation model.
    It aims to predict the CTR given a set of features by using logistic regression,
    which is ideally suited for probabilities as it always predicts a value between 0 and 1:

    .. math::
        CTR = \frac{1}{1+e^{-Z}}

        Z = \sum_{i} {w_i}{x_i}
    """
    def __init__(self, config, dataset):
        super(LR, self).__init__(config, dataset)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
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
