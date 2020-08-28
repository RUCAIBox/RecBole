# -*- coding: utf-8 -*-
# @Time   : 2020/7/16 11:25
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.con
# @File   : NGCF.py

"""
Reference:
Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering.
In SIGIR'19, Paris, France, July 21-25, 2019.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from ...utils import InputType
from ..abstract_recommender import GeneralRecommender
from ..layers import BiGNNLayer


class NGCF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NGCF, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.embedding_size = config['embedding_size']
        self.layers = config['layers']
        self.layers = [self.embedding_size] + self.layers
        self.node_dropout = config['node_dropout']
        self.message_dropout = config['message_dropout']
        self.device = config['device']
        self.delay = config['delay']
        self.batch_size = config['train_batch_size']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = torch.nn.ModuleList()
        for From, To in zip(self.layers[:-1], self.layers[1:]):
            self.GNNlayers.append(BiGNNLayer(From, To))
        self.sigmoid = nn.LogSigmoid()
        self.apply(self.init_weights)
        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        self.restore_user_e = None
        self.restore_item_e = None

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

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

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def get_feature_matrix(self):
        user_embd = self.user_embedding.weight
        item_embd = self.item_embedding.weight
        features = torch.cat([user_embd, item_embd], dim=0)
        return features

    def forward(self):
        A_hat = self.sparse_dropout(self.norm_adj_matrix,
                                    self.node_dropout,
                            self.norm_adj_matrix._nnz()) if self.node_dropout != 0 else self.norm_adj_matrix
        features = self.get_feature_matrix()
        finalEmbd = [features.clone()]
        for gnn in self.GNNlayers:
            features = gnn(A_hat, self.eye_matrix, features)
            features = nn.LeakyReLU(negative_slope=0.2)(features)
            features = nn.Dropout(self.message_dropout)(features)
            features = F.normalize(features, p=2, dim=1)
            finalEmbd += [features.clone()]
        finalEmbd = torch.cat(finalEmbd, dim=1)

        u_g_embeddings = finalEmbd[:self.n_users, :]
        i_g_embeddings = finalEmbd[self.n_users:, :]

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        u_embedding, i_embedding = self.forward()
        u_embeddings = u_embedding[user, :]
        posi_embeddings = i_embedding[pos_item, :]
        negi_embeddings = i_embedding[neg_item, :]

        pos_scores = torch.sum(torch.mul(u_embeddings, posi_embeddings), axis=1)
        neg_scores = torch.sum(torch.mul(u_embeddings, negi_embeddings), axis=1)

        maxi = self.sigmoid(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = torch.norm(u_embeddings, p=2)+torch.norm(posi_embeddings, p=2)+torch.norm(negi_embeddings, p=2)
        emb_loss = self.delay * regularizer / self.batch_size

        return mf_loss + emb_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        u_embedding, i_embedding = self.forward()

        u_embeddings = u_embedding[user, :]
        i_embeddings = i_embedding[item, :]
        scores = torch.sum(torch.mul(u_embeddings, i_embeddings), axis=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user, :]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

