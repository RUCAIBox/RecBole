# -*- coding: utf-8 -*-
# @Time   : 2020/10/6
# @Author : Yingqian Min
# @Email  : eliver_min@foxmail.com

r"""
ConvNCF
################################################
Reference:
    Xiangnan He et al. "Outer Product-based Neural Collaborative Filtering." in IJCAI 2018.

Reference code:
    https://github.com/duxy-me/ConvNCF
"""

import torch
import torch.nn as nn

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers, CNNLayers


class ConvNCFBPRLoss(nn.Module):

    """ ConvNCFBPRLoss, based on Bayesian Personalized Ranking,
    
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = ConvNCFBPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """
    def __init__(self):
        super(ConvNCFBPRLoss, self).__init__()
    
    def forward(self, pos_score, neg_score):
        distance = pos_score - neg_score
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))
        return loss


class ConvNCF(GeneralRecommender):
    r"""ConvNCF is a a new neural network framework for collaborative filtering based on NCF.
    It uses an outer product operation above the embedding layer, 
    which results in a semantic-rich interaction map that encodes pairwise correlations between embedding dimensions.
    We carefully design the data interface and use sparse tensor to train and test efficiently.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ConvNCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.cnn_channels = config['cnn_channels']
        self.cnn_kernels = config['cnn_kernels']
        self.cnn_strides = config['cnn_strides']
        self.dropout_prob = config['dropout_prob']
        self.regs = config['reg_weights']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.cnn_layers = CNNLayers(self.cnn_channels, self.cnn_kernels, self.cnn_strides, activation='relu')
        self.predict_layers = MLPLayers([self.cnn_channels[-1], 1], self.dropout_prob, activation='none')
        self.loss = ConvNCFBPRLoss()

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        interaction_map = torch.bmm(user_e.unsqueeze(2), item_e.unsqueeze(1))
        interaction_map = interaction_map.unsqueeze(1)

        cnn_output = self.cnn_layers(interaction_map)
        cnn_output = cnn_output.sum(axis=(2, 3))

        prediction = self.predict_layers(cnn_output)
        prediction = prediction.squeeze()

        return prediction

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrixes and weight matrixes of model.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_1, reg_2 = self.regs[:2]
        loss_1 = reg_1 * self.user_embedding.weight.norm(2)
        loss_2 = reg_1 * self.item_embedding.weight.norm(2)
        loss_3 = 0
        for name, parm in self.cnn_layers.named_parameters():
            if name.endswith('weight'):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        for name, parm in self.predict_layers.named_parameters():
            if name.endswith('weight'):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        
        loss = self.loss(pos_item_score, neg_item_score)
        opt_loss = loss + self.reg_loss()

        return opt_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)
