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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from model.abstract_recommender import AbstractRecommender
from model.layers import BiGNNLayer
import numpy as np
import scipy.sparse as sp


class NgCf(AbstractRecommender):

    def __init__(self, config, dataset):
        super(NgCf, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.LABEL = config['LABEL_FIELD']
        self.n_users = len(dataset.field2id_token[self.USER_ID])
        self.n_items = len(dataset.field2id_token[self.ITEM_ID])
        self.embedding_size = config['embedding_size']
        self.layers = config['layers']
        self.layers = [self.embedding_size] + self.layers
        self.node_dropout = config['node_dropout']
        self.message_dropout = config['message_dropout']
        self.device = config['device']
        self.delay = config['delay']
        self.batch_size = config['train_batch_size']
        self.interaction_matrix = dataset.train_matrix.tocsr().astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = torch.nn.ModuleList()
        for From, To in zip(self.layers[:-1], self.layers[1:]):
            self.GNNlayers.append(BiGNNLayer(From, To))

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self.init_weights)

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        A = A.tolil()
        A[:self.num_users, self.num_users:] = self.interaction_matrix
        A[self.num_users:, :self.num_users] = self.interaction_matrix.transpose()
        A = A.todok()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
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
        i = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
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
        uidx = torch.LongTensor([i for i in range(self.userNum)]).to(self.device)
        iidx = torch.LongTensor([i for i in range(self.itemNum)]).to(self.device)

        userEmbd = self.user_embedding(uidx)
        itemEmbd = self.item_embedding(iidx)
        features = torch.cat([userEmbd, itemEmbd], dim=0)
        return features

    def forward(self):
        A_hat = self.sparse_dropout(self.norm_adj_matrix,
                                    self.node_dropout,
                            self.norm_adj_matrix._nnz()) if self.node_dropout is not None else self.norm_adj_matrix
        features = self.get_feature_matrix()
        finalEmbd = [features.clone()]
        for gnn in self.GNNlayers:
            features = gnn(A_hat, self.eye_matrix, features)
            features = nn.LeakyReLU(negative_slope=0.2)(features)
            features = nn.Dropout(self.message_dropout)(features)
            features = F.normalize(features, p=2, dim=1)
            finalEmbd += [features.clone()]
        finalEmbd = torch.cat(finalEmbd, dim=1)

        u_g_embeddings = finalEmbd[:self.n_user, :]
        i_g_embeddings = finalEmbd[self.n_user:, :]

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        u_embedding, i_embedding = self.forward()
        u_embeddings = u_embedding[user, :]
        posi_embeddings = i_embedding[pos_item, :]
        negi_embeddings = i_embedding[neg_item, :]

        pos_scores = torch.sum(torch.mul(u_embeddings, posi_embeddings), axis=1)
        neg_scores = torch.sum(torch.mul(u_embeddings, negi_embeddings), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(u_embeddings) ** 2
                       + torch.norm(posi_embeddings) ** 2
                       + torch.norm(negi_embeddings) ** 2) / 2
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
