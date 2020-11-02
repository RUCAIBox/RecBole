# -*- coding: utf-8 -*-
# @Time   : 2020/6/27
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/22,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
NeuMF
################################################
Reference:
    Xiangnan He et al. "Neural Collaborative Filtering." in WWW 2017.
"""

import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers


class NeuMF(GeneralRecommender):
    r"""NeuMF is an neural network enhanced matrix factorization model.
    It replace the dot product to mlp for a more precise user-item interaction.

    Note:

        Our implementation only contains a rough pretraining function.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.use_pretrain = config['use_pretrain']
        self.mf_pretrain_path = config['mf_pretrain_path']
        self.mlp_pretrain_path = config['mlp_pretrain_path']

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers([2 * self.mlp_embedding_size] + self.mlp_hidden_size)
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], 1, self.dropout_prob)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters.

        """
        mf = torch.load(self.mf_pretrain_path)
        mlp = torch.load(self.mlp_pretrain_path)
        self.user_mf_embedding.weight.data.copy_(mf.user_mf_embedding.weight)
        self.item_mf_embedding.weight.data.copy_(mf.item_mf_embedding.weight)
        self.user_mlp_embedding.weight.data.copy_(mlp.user_mlp_embedding.weight)
        self.item_mlp_embedding.weight.data.copy_(mlp.item_mlp_embedding.weight)

        for (m1, m2) in zip(self.mlp_layers.mlp_layers, mlp.mlp_layers.mlp_layers):
            if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                m1.weight.data.copy_(m2.weight)
                m1.bias.data.copy_(m2.bias)

        predict_weight = torch.cat([mf.predict_layer.weight,
                                    mlp.predict_layer.weight], dim=1)
        predict_bias = mf.predict_layer.bias + mlp.predict_layer.bias

        self.predict_layer.weight.data.copy_(0.5 * predict_weight)
        self.predict_layer.weight.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)     # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))   # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        elif self.mf_train:
            output = self.sigmoid(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = self.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')
        return output.squeeze()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain.

        """
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)
