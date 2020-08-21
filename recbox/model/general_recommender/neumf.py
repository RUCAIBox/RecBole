# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 15:10
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : neumf.py

"""
Reference:
Xiangnan He et al., "Neural Collaborative Filtering." in WWW 2017.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from ...utils import InputType
from ..abstract_recommender import GeneralRecommender
from ..layers import MLPLayers


class NeuMF(GeneralRecommender):

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__()
        self.input_type = InputType.POINTWISE
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.LABEL = config['LABEL_FIELD']
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout = config['dropout']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.use_pretrain = config['use_pretrain']

        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers([2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout)
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        if self.use_pretrain:
            mf = torch.load('./saved/NeuMF-mf.pth')
            mlp = torch.load('./saved/NeuMF-mlp.pth')
            self.load_state_dict(mf, strict=False)
            self.load_state_dict(mlp, strict=False)
        else:
            self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)     # (batch, embedding_size)
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))   # (batch, layers[-1])
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

    def dump_paramaters(self):
        full_dict = self.state_dict()
        target_dict = dict()
        for name in full_dict:
            if self.mf_train and 'mf' in name:
                target_dict[name] = full_dict[name]
            if self.mlp_train and 'mlp' in name:
                target_dict[name] = full_dict[name]
        if self.mf_train:
            save_path = './saved/NeuMF-mf.pth'
        elif self.mlp_train:
            save_path = './saved/NeuMF-mlp.pth'
        else:
            save_path = './saved/NeuMF.pth'
        torch.save(target_dict, save_path)
