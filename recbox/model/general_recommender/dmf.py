# _*_ coding: utf-8 _*_
# @CreateTime : 2020/8/21 16:58 
# @Author : Kaizhou Zhang
# @Email  : kaizhou361@163.com
# @File : dmf.py

"""
Reference:
Hong-Jian Xue et al., "Deep Matrix Factorization Models for Recommender Systems." in IJCAI 2017.
"""

import torch
import torch.nn as nn
import numpy as np

from ..abstract_recommender import GeneralRecommender
from ...utils import InputType
from ..layers import MLPLayers

class DMF(GeneralRecommender):

    def __init__(self, config, dataset):
        super(DMF, self).__init__()
        self.input_type = InputType.POINTWISE
        self.device = config['device']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.layers = config['layers']
        self.latent_dim = self.layers[0]
        self.min_y_hat = config['min_y_hat']
        self.interaction_matrix = None
        self.u_embedding = None
        self.i_embedding = None

        self.linear_user = nn.Linear(in_features=self.n_items, out_features=self.latent_dim,bias=False)
        self.linear_user.weight.detach().normal_(0, 0.01)
        self.linear_item = nn.Linear(in_features=self.n_users, out_features=self.latent_dim,bias=False)
        self.linear_item.weight.detach().normal_(0, 0.01)

        self.user_fc_layers = MLPLayers(self.layers)
        self.item_fc_layers = MLPLayers(self.layers)

    def train_preparation(self,train_data,valid_data):
        self.interaction_matrix = train_data.inter_matrix(form='csr').astype(np.float32)

    def forward(self, user, item):
        user = torch.from_numpy(self.interaction_matrix[user.cpu()].todense()).to(self.device)
        item = torch.from_numpy(self.interaction_matrix[:, item.cpu()].todense()).to(self.device).t()
        user = self.linear_user(user)
        item = self.linear_item(item)

        user = self.user_fc_layers(user)
        item = self.item_fc_layers(item)

        vector = torch.cosine_similarity(user, item).view(-1,)
        vector = torch.max(vector, torch.tensor([self.min_y_hat]).to(self.device))
        return vector

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # print(user)
        output = self.forward(user,item)
        self.max_rating = self.interaction_matrix.max()
        reg_rating = output / self.max_rating
        loss = (reg_rating * (output.log()) + (1 - reg_rating) * ((1 - output).log())).mean()
        loss = -loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user,item)

    def get_user_embedding(self, user):
        user = torch.from_numpy(self.interaction_matrix[user.cpu()].todense()).to(self.device)
        user = self.linear_user(user)
        user = self.user_fc_layers(user)
        return user

    def get_item_embedding(self):
        interaction_matrix = self.interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(interaction_matrix.data)
        item_matrix = torch.sparse.FloatTensor(i, data).to(self.device).transpose(0, 1)

        item = torch.sparse.mm(item_matrix, self.linear_item.weight.t())

        item = self.item_fc_layers(item)
        return item

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        self.u_embedding = self.get_user_embedding(user)
        if self.i_embedding is None:
            self.i_embedding = self.get_item_embedding()
        u_sqrt = torch.mul(self.u_embedding, self.u_embedding).sum(dim=1).sqrt().view(-1,1)
        i_sqrt = torch.mul(self.i_embedding, self.i_embedding).sum(dim=1).sqrt().view(1,-1)
        cos_similarity = torch.mm(self.u_embedding, self.i_embedding.t()) / torch.mm(u_sqrt, i_sqrt)
        cos_similarity = torch.max(cos_similarity, torch.tensor([self.min_y_hat]).to(self.device))
        return cos_similarity.view(-1)