# -*- coding: utf-8 -*-
# @Time   : 2020/8/29 10:30
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.con
# @File   : lightgcn.py

"""
Reference:
He, Xiangnan, et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation."
arXiv preprint arXiv:2002.02126 (2020)..
"""

# dataset=ml-1m
# Parameters: {'delay': 0.001, 'layers': 2, 'learning_rate': 0.002}
# test result: {'Recall@10': 0.1713, 'MRR@10': 0.3738, 'NDCG@10': 0.2118, 'Hit@10': 0.725, 'Precision@10': 0.1569}
#
# dataset=yelp
# Parameters: {'delay': 1e-4, 'layers': 3, 'learning_rate': 0.001}
# test result: {'Recall@10': 0.0597, 'MRR@10': 0.0728, 'NDCG@10': 0.0458, 'Hit@10': 0.1867, 'Precision@10': 0.023}

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import InputType
from ..abstract_recommender import GeneralRecommender
from ..loss import BPRLoss


class LightGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.latent_dim = config['embedding_size']  # int:the embedding size of lightGCN
        self.n_layers = config['layers']  # int:the layer num of lightGCN
        self.weight_decay = config['delay']  # the weight decay for l2 normalizaton
        self.device = config['device']

        self.dataset = dataset
        self.num_users = dataset.num(self.USER_ID)
        self.num_items = dataset.num(self.ITEM_ID)

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        # print('using xavier initilizer')

        self.Sigmoid = nn.Sigmoid()
        self.BPRLoss = BPRLoss(gamma=self.weight_decay)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32) # csr
        self.Graph = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        self.restore_user_e = None
        self.restore_item_e = None

        # future work
        self.dropout_on = False
        self.keep_prob = 0.6  # dropout prob
        self.A_split = False
        # print(f"lgn is already to go(dropout:{self.dropout_on})")

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        A = A.tolil()
        A[:self.num_users, self.num_users:] = self.interaction_matrix
        A[self.num_users:, :self.num_users] = self.interaction_matrix.transpose()
        A = A.todok()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7  # add epsilon to avoid Devide by zero Warning
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
        num = self.num_items + self.num_users
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # loss = - self.BPRLoss(pos_scores, neg_scores)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        return loss, reg_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        loss, reg_loss = self.bpr_loss(user, pos_item, neg_item)
        return loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        scores = self.forward(user, item)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.computer()
        u_embeddings = self.restore_user_e[user, :]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)