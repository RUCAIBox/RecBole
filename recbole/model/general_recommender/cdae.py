# -*- coding: utf-8 -*-
# @Time   : 2020/12/12
# @Author : Xingyu Pan
# @Email  : panxy@ruc.edu.cn

r"""
CDAE
################################################
Reference:
    Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. In WSDM 2016.

Reference code:
    https://github.com/jasonyaw/CDAE
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class CDAE(GeneralRecommender, AutoEncoderMixin):
    r"""Collaborative Denoising Auto-Encoder (CDAE) is a recommendation model
    for top-N recommendation that utilizes the idea of Denoising Auto-Encoders.
    We implement the the CDAE model with only user dataloader.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CDAE, self).__init__(config, dataset)

        self.reg_weight_1 = config["reg_weight_1"]
        self.reg_weight_2 = config["reg_weight_2"]
        self.loss_type = config["loss_type"]
        self.hid_activation = config["hid_activation"]
        self.out_activation = config["out_activation"]
        self.embedding_size = config["embedding_size"]
        self.corruption_ratio = config["corruption_ratio"]

        self.build_histroy_items(dataset)

        if self.hid_activation == "sigmoid":
            self.h_act = nn.Sigmoid()
        elif self.hid_activation == "relu":
            self.h_act = nn.ReLU()
        elif self.hid_activation == "tanh":
            self.h_act = nn.Tanh()
        else:
            raise ValueError("Invalid hidden layer activation function")

        if self.out_activation == "sigmoid":
            self.o_act = nn.Sigmoid()
        elif self.out_activation == "relu":
            self.o_act = nn.ReLU()
        else:
            raise ValueError("Invalid output layer activation function")

        self.dropout = nn.Dropout(p=self.corruption_ratio)

        self.h_user = nn.Embedding(self.n_users, self.embedding_size)
        self.h_item = nn.Linear(self.n_items, self.embedding_size)
        self.out_layer = nn.Linear(self.embedding_size, self.n_items)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, x_items, x_users):
        h_i = self.dropout(x_items)
        h_i = self.h_item(h_i)
        h_u = self.h_user(x_users)
        h = torch.add(h_u, h_i)
        h = self.h_act(h)
        out = self.out_layer(h)
        return out

    def calculate_loss(self, interaction):
        x_users = interaction[self.USER_ID]
        x_items = self.get_rating_matrix(x_users)
        predict = self.forward(x_items, x_users)

        if self.loss_type == "MSE":
            predict = self.o_act(predict)
            loss_func = nn.MSELoss(reduction="sum")
        elif self.loss_type == "BCE":
            loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise ValueError("Invalid loss_type, loss_type must in [MSE, BCE]")
        loss = loss_func(predict, x_items)
        # l1-regularization
        loss += self.reg_weight_1 * (
            self.h_user.weight.norm(p=1) + self.h_item.weight.norm(p=1)
        )
        # l2-regularization
        loss += self.reg_weight_2 * (
            self.h_user.weight.norm() + self.h_item.weight.norm()
        )

        return loss

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        predict_items = interaction[self.ITEM_ID]

        items = self.get_rating_matrix(users)
        scores = self.forward(items, users)
        scores = self.o_act(scores)
        return scores[[torch.arange(len(predict_items)).to(self.device), predict_items]]

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]

        items = self.get_rating_matrix(users)
        predict = self.forward(items, users)
        predict = self.o_act(predict)
        return predict.view(-1)
