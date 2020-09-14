# -*- coding: utf-8 -*-
# @Time   : 2020/7/16 11:25
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.con
# @File   : NGCF.py

"""
Reference:
Xiang Wang et al. "Neural Graph Collaborative Filtering." in SIGIR 2019.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import InputType
from ..abstract_recommender import GeneralRecommender
from ..loss import BPRLoss, EmbLoss
from ..layers import BiGNNLayer
from ..utils import xavier_normal_initialization


def sparse_dropout(x, rate, noise_shape):
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    return out * (1. / (1 - rate))


class NGCF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NGCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.layers = config['layers']
        self.layers = [self.embedding_size] + self.layers
        self.node_dropout = config['node_dropout']
        self.message_dropout = config['message_dropout']
        self.reg_weight = config['reg_weight']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        A = A.tolil()
        A[:self.n_users, self.n_users:] = self.interaction_matrix
        A[self.n_users:, :self.n_users] = self.interaction_matrix.transpose()
        A = A.todok()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7     # add epsilon to avoid Devide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data)
        return SparseL

    def get_eye_mat(self):
        num = self.n_items + self.n_users
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        A_hat = sparse_dropout(self.norm_adj_matrix, self.node_dropout,
                               self.norm_adj_matrix._nnz()) if self.node_dropout != 0 else self.norm_adj_matrix
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        posi_embeddings = item_all_embeddings[pos_item]
        negi_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.sum(torch.mul(u_embeddings, posi_embeddings), axis=1)
        neg_scores = torch.sum(torch.mul(u_embeddings, negi_embeddings), axis=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        reg_loss = self.reg_loss([u_embeddings, posi_embeddings, negi_embeddings])

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.sum(torch.mul(u_embeddings, i_embeddings), axis=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
