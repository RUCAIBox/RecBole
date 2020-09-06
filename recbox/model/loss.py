# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


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


class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


# todo: wait to be test
class MarginLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_score, neg_score):
        dev = pos_score.device
        cache_zeros = torch.zeros_like(pos_score).to(dev)

        loss = torch.sum(torch.max(pos_score - neg_score + self.margin, cache_zeros))
        return loss
