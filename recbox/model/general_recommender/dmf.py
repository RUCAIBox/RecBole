# _*_ coding: utf-8 _*_
# @CreateTime : 2020/8/21 16:58 
# @Author : Kaizhou Zhang
# @Email  : kaizhou361@163.com
# @File : dmf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..abstract_recommender import GeneralRecommender
from ...utils import InputType

class DMF(GeneralRecommender):

    def __init__(self,config,dataset):
        super(DMF, self).__init__()
        self.input_type = InputType.POINTWISE
        self.device = config['device']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.rating_field = config['RATING_FIELD']
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.layers = config['layers']
        self.latent_dim = self.layers[0]
        self.interaction_matrix = None

        self.linear_user_1 = nn.Linear(in_features=self.n_items, out_features=self.latent_dim)
        self.linear_user_1.weight.detach().normal_(0, 0.01)
        self.linear_item_1 = nn.Linear(in_features=self.n_users, out_features=self.latent_dim)
        self.linear_item_1.weight.detach().normal_(0, 0.01)

        self.user_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.user_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))

        self.item_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.item_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))

    def train_preparation(self,train_data,valid_data):
        self.interaction_matrix = train_data.inter_matrix(form='csr',value_field=self.rating_field).astype(np.float32)
        self.interaction_matrix = torch.from_numpy(self.interaction_matrix.toarray()).to(self.device)
        #print('111')

    def forward(self, user, item):
        user = self.linear_user_1(user)
        item = self.linear_item_1(item)

        for idx in range(len(self.layers) - 1):
            user = F.relu(user)
            user = self.user_fc_layers[idx](user)

        for idx in range(len(self.layers) - 1):
            item = F.relu(item)
            item = self.item_fc_layers[idx](item)

        vector = torch.cosine_similarity(user, item).view(-1, 1)  # 得到一个列tensor
        vector = torch.clamp(vector, min=1e-6, max=1)

        return vector

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user = self.interaction_matrix[user]
        item = self.interaction_matrix[:, item].t()
        output = self.forward(user,item)
        self.max_rating = self.interaction_matrix.max()
        regRate = output / self.max_rating
        loss = (regRate * (output.log()) + (1 - regRate) * ((1 - output).log())).mean()
        loss = -loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user = self.interaction_matrix[user]
        item = self.interaction_matrix[:, item].t()
        return self.forward(user,item)