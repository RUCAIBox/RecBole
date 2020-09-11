# _*_ coding: utf-8 _*_
# @CreateTime : 2020/8/21 16:58
# @Author : Kaizhou Zhang
# @Email  : kaizhou361@163.com
# @File : dmf.py

# UPDATE
# @Time    :   2020/08/31
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

"""
Reference:
Hong-Jian Xue et al., "Deep Matrix Factorization Models for Recommender Systems." in IJCAI 2017.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from ...utils import InputType
from ..abstract_recommender import GeneralRecommender
from ..layers import MLPLayers


class DMF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DMF, self).__init__()
        self.device = config['device']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']

        self.user_layers_dim = config['user_layers_dim']
        self.item_layers_dim = config['item_layers_dim']
        assert self.user_layers_dim[-1] == self.item_layers_dim[-1], 'The dimensions of the last layer of users and items must be the same'

        self.min_y_hat = config['min_y_hat']
        self.inter_matrix_type = config['inter_matrix_type']
        if self.inter_matrix_type == '01':
            self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        elif self.inter_matrix_type == 'rating':
            self.interaction_matrix = dataset.inter_matrix(form='csr', value_field=self.RATING).astype(np.float32)
        else:
            raise ValueError("The inter_matrix_type must in ['01', 'rating'] but get {}".format(self.inter_matrix_type))
        self.max_rating = self.interaction_matrix.max()

        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        self.user_linear = nn.Linear(in_features=self.n_items, out_features=self.user_layers_dim[0], bias=False)
        self.item_linear = nn.Linear(in_features=self.n_users, out_features=self.item_layers_dim[0], bias=False)

        self.user_fc_layers = MLPLayers(self.user_layers_dim)
        self.item_fc_layers = MLPLayers(self.item_layers_dim)
        self.apply(self.init_weights)

        # Save the item embedding before dot product layer to speed up evaluation
        self.i_embedding = None

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, user, item):
        user = torch.from_numpy(self.interaction_matrix[user.cpu()].todense()).to(self.device)
        item = torch.from_numpy(self.interaction_matrix[:, item.cpu()].todense()).to(self.device).t()
        user = self.user_linear(user)
        item = self.item_linear(item)

        user = self.user_fc_layers(user)
        item = self.item_fc_layers(item)

        vector = torch.cosine_similarity(user, item).view(-1)
        vector = torch.max(vector, torch.tensor([self.min_y_hat]).to(self.device))
        return vector

    def calculate_loss(self, interaction):
        # when starting a new epoch, the item embedding we saved must be cleared
        if self.training:
            self.i_embedding = None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.inter_matrix_type == '01':
            label = interaction[self.LABEL]
        elif self.inter_matrix_type == 'rating':
            label = interaction[self.RATING] * interaction[self.LABEL]
        output = self.forward(user, item)
        label = label / self.max_rating
        loss = -(label * (output.log()) + (1 - label) * ((1 - output).log())).mean()
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def get_user_embedding(self, user):
        user = torch.from_numpy(self.interaction_matrix[user.cpu()].todense()).to(self.device)
        user = self.user_linear(user)
        user = self.user_fc_layers(user)
        return user

    def get_item_embedding(self):
        interaction_matrix = self.interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(interaction_matrix.data)
        item_matrix = torch.sparse.FloatTensor(i, data).to(self.device).transpose(0, 1)

        item = torch.sparse.mm(item_matrix, self.item_linear.weight.t())
        item = self.item_fc_layers(item)
        return item

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_embedding = self.get_user_embedding(user)
        u_embedding = F.normalize(u_embedding, p=2, dim=1)

        if self.i_embedding is None:
            self.i_embedding = self.get_item_embedding()
            self.i_embedding = F.normalize(self.i_embedding, p=2, dim=1)

        cos_similarity = torch.mm(u_embedding, self.i_embedding.t())
        cos_similarity = torch.max(cos_similarity, torch.tensor([self.min_y_hat]).to(self.device))
        return cos_similarity.view(-1)
