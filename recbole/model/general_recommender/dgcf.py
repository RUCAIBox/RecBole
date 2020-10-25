# -*- coding: utf-8 -*-
# @Time   : 2020/8/26
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
DGCF
################################################
Reference:
    Wang Xiang et al. "Disentangled Graph Collaborative Filtering." in SIGIR 2020.

Reference code:
    https://github.com/xiangwang1223/disentangled_graph_collaborative_filtering
"""

import numpy as np
import random as rd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_normal_initialization


def sample_cor_samples(n_users, n_items, cor_batch_size):
    r"""This is a function that sample item ids and user ids.

    Args:
        n_users (int): number of users in total
        n_items (int): number of items in total
        cor_batch_size (int): number of id to sample

    Returns:
        list: cor_users, cor_items. The result sampled ids with both as cor_batch_size long.

    Note:
        We have to sample some embedded representations out of all nodes.
        Becasue we have no way to store cor-distance for each pair.
    """
    cor_users = rd.sample(list(range(n_users)), cor_batch_size)
    cor_items = rd.sample(list(range(n_items)), cor_batch_size)

    return cor_users, cor_items


class DGCF(GeneralRecommender):
    r"""DGCF is a disentangled representation enhanced matrix factorization model.
    The interaction matrix of :math:`n_{users} \times n_{items}` is decomposed to :math:`n_{factors}` intent graph,
    we carefully design the data interface and use sparse tensor to train and test efficiently.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DGCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_factors = config['n_factors']
        self.n_iterations = config['n_iterations']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cor_weight = config['cor_weight']
        n_batch = dataset.dataset.inter_num // self.batch_size + 1
        self.cor_batch_size = int(max(self.n_users / n_batch, self.n_items / n_batch))
        # ensure embedding can be divided into <n_factors> intent
        assert self.embedding_size % self.n_factors == 0

        # generate intermediate data
        row = self.interaction_matrix.row.tolist()
        col = self.interaction_matrix.col.tolist()
        col = [item_index + self.n_users for item_index in col]
        all_h_list = row + col  # row.extend(col)
        all_t_list = col + row  # col.extend(row)
        num_edge = len(all_h_list)
        edge_ids = range(num_edge)
        self.all_h_list = torch.LongTensor(all_h_list).to(self.device)
        self.all_t_list = torch.LongTensor(all_t_list).to(self.device)
        self.edge2head = torch.LongTensor([all_h_list, edge_ids]).to(self.device)
        self.head2edge = torch.LongTensor([edge_ids, all_h_list]).to(self.device)
        self.tail2edge = torch.LongTensor([edge_ids, all_t_list]).to(self.device)
        val_one = torch.ones_like(self.all_h_list).float().to(self.device)
        num_node = self.n_users + self.n_items
        self.edge2head_mat = self._build_sparse_tensor(self.edge2head, val_one, (num_node, num_edge))
        self.head2edge_mat = self._build_sparse_tensor(self.head2edge, val_one, (num_edge, num_node))
        self.tail2edge_mat = self._build_sparse_tensor(self.tail2edge, val_one, (num_edge, num_node))
        self.num_edge = num_edge
        self.num_node = num_node

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _build_sparse_tensor(self, indices, values, size):
        # Construct the sparse matrix with indices, values and size.
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)

    def _get_ego_embeddings(self):
        # concat of user embeddings and item embeddings
        user_embd = self.user_embedding.weight
        item_embd = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embd, item_embd], dim=0)
        return ego_embeddings

    def build_matrix(self, A_values):
        r"""Get the normalized interaction matrix of users and items according to A_values.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        Args:
            A_values (torch.cuda.FloatTensor): (num_edge, n_factors)

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            torch.cuda.FloatTensor: Sparse tensor of the normalized interaction matrix. shape: (num_edge, n_factors)
        """
        norm_A_values = self.softmax(A_values)
        factor_edge_weight = []
        for i in range(self.n_factors):
            tp_values = norm_A_values[:, i].unsqueeze(1)
            # (num_edge, 1)
            d_values = torch.sparse.mm(self.edge2head_mat, tp_values)
            # (num_node, num_edge) (num_edge, 1) -> (num_node, 1)
            d_values = torch.clamp(d_values, min=1e-8)
            try:
                assert not torch.isnan(d_values).any()
            except AssertionError:
                print("d_values", torch.min(d_values), torch.max(d_values))

            d_values = 1.0 / torch.sqrt(d_values)
            head_term = torch.sparse.mm(self.head2edge_mat, d_values)
            # (num_edge, num_node) (num_node, 1) -> (num_edge, 1)

            tail_term = torch.sparse.mm(self.tail2edge_mat, d_values)
            edge_weight = tp_values * head_term * tail_term
            factor_edge_weight.append(edge_weight)
        return factor_edge_weight

    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        all_embeddings = [ego_embeddings.unsqueeze(1)]
        # initialize with every factor value as 1
        A_values = torch.ones((self.num_edge, self.n_factors)).to(self.device)
        A_values = Variable(A_values, requires_grad=True)
        for k in range(self.n_layers):
            layer_embeddings = []

            # split the input embedding table
            # .... ego_layer_embeddings is a (n_factors)-leng list of embeddings [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = torch.chunk(ego_embeddings, self.n_factors, 1)
            for t in range(0, self.n_iterations):
                iter_embeddings = []
                A_iter_values = []
                factor_edge_weight = self.build_matrix(A_values=A_values)
                for i in range(0, self.n_factors):
                    # update the embeddings via simplified graph convolution layer
                    edge_weight = factor_edge_weight[i]
                    # (num_edge, 1)
                    edge_val = torch.sparse.mm(self.tail2edge_mat, ego_layer_embeddings[i])
                    # (num_edge, dim / n_factors)
                    edge_val = edge_val * edge_weight
                    # (num_edge, dim / n_factors)
                    factor_embeddings = torch.sparse.mm(self.edge2head_mat, edge_val)
                    # (num_node, num_edge) (num_edge, dim) -> (num_node, dim)

                    iter_embeddings.append(factor_embeddings)

                    if t == self.n_iterations - 1:
                        layer_embeddings = iter_embeddings

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    head_factor_embedings = torch.index_select(factor_embeddings, dim=0, index=self.all_h_list)
                    tail_factor_embedings = torch.index_select(ego_layer_embeddings[i], dim=0, index=self.all_t_list)

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    # to adapt to torch version
                    head_factor_embedings = F.normalize(head_factor_embedings, p=2, dim=1)
                    tail_factor_embedings = F.normalize(tail_factor_embedings, p=2, dim=1)

                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [num_edge, 1]
                    A_factor_values = torch.sum(head_factor_embedings * torch.tanh(tail_factor_embedings),
                                                dim=1, keepdim=True)

                    # update the attentive weights
                    A_iter_values.append(A_factor_values)
                A_iter_values = torch.cat(A_iter_values, dim=1)
                # (num_edge, n_factors)
                # add all layer-wise attentive weights up.
                A_values = A_values + A_iter_values

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = torch.cat(layer_embeddings, dim=1)

            ego_embeddings = side_embeddings
            # concatenate outputs of all layers
            all_embeddings += [ego_embeddings.unsqueeze(1)]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        # (num_node, n_layer + 1, embedding_size)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        # (num_node, embedding_size)

        u_g_embeddings = all_embeddings[:self.n_users, :]
        i_g_embeddings = all_embeddings[self.n_users:, :]

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        posi_embeddings = item_all_embeddings[pos_item]
        negi_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # cul regularizer
        u_ego_embeddings = self.user_embedding(user)
        posi_ego_embeddings = self.item_embedding(pos_item)
        negi_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)

        if self.n_factors > 1 and self.cor_weight > 1e-9:
            cor_users, cor_items = sample_cor_samples(self.n_users, self.n_items, self.cor_batch_size)
            cor_users = torch.LongTensor(cor_users).to(self.device)
            cor_items = torch.LongTensor(cor_items).to(self.device)
            cor_u_embeddings = user_all_embeddings[cor_users]
            cor_i_embeddings = item_all_embeddings[cor_items]
            cor_loss = self.create_cor_loss(cor_u_embeddings, cor_i_embeddings)
            loss = mf_loss + self.reg_weight * reg_loss + self.cor_weight * cor_loss
        else:
            loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        r"""Calculate the correlation loss for a sampled users and items.

        Args:
            cor_u_embeddings (torch.cuda.FloatTensor): (cor_batch_size, n_factors)
            cor_i_embeddings (torch.cuda.FloatTensor): (cor_batch_size, n_factors)

        Returns:
            torch.Tensor : correlation loss.
        """
        cor_loss = None

        ui_embeddings = torch.cat((cor_u_embeddings, cor_i_embeddings), dim=0)
        ui_factor_embeddings = torch.chunk(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors - 1):
            x = ui_factor_embeddings[i]
            # (M + N, emb_size / n_factor)
            y = ui_factor_embeddings[i + 1]
            # (M + N, emb_size / n_factor)
            if i == 0:
                cor_loss = self._create_distance_correlation(x, y)
            else:
                cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors / 2)

        return cor_loss

    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
            X: (batch_size, dim)
            return: X - E(X)
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            r = torch.sum(X * X, dim=1, keepdim=True)
            # (N, 1)
            # (x^2 - 2xy + y^2) -> l2 distance between all vectors
            value = r - 2 * torch.mm(X, X.T + r.T)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            D = torch.sqrt(value + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            # matrix - average over row - average over col + average over matrix
            D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = float(D1.size(0))
            value = torch.sum(D1 * D2) / (n_samples * n_samples)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            dcov = torch.sqrt(value + 1e-8)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        dcor = dcov_12 / (torch.sqrt(value) + 1e-10)
        return dcor

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        u_embedding, i_embedding = self.forward()

        u_embeddings = u_embedding[user]
        i_embeddings = i_embedding[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
