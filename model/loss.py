# -*- coding: utf-8 -*-
# @Time   : 2020/6/26 16:41
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : loss.py

"""
Common Loss in recommender system
"""


import torch
import torch.nn as nn
import torch.nn.functional as fn


class BPRLoss(nn.Module):

    """ BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float):

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

     Examples::

        >> loss = BPRLoss()
        >> pos_score = torch.randn(3, requires_grad=True)
        >> neg_score = torch.randn(3, requires_grad=True)
        >> output = loss(pos_score, neg_score)
        >> output.backward()
    """
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
