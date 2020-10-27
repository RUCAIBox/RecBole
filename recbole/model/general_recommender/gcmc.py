# -*- coding: utf-8 -*-
# @Time   : 2020/9/1 14:00
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE
# @Time   : 2020/10/1
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

r"""
GCMC
################################################

Reference:
    van den Berg et al. "Graph Convolutional Matrix Completion." in SIGKDD 2018.

Reference code:
    https://github.com/riannevdberg/gc-mc
"""


import math
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


class GCMC(GeneralRecommender):
    r"""GCMC is a model that incorporate graph autoencoders for recommendation.

    Graph autoencoders are comprised of: 

    1) a graph encoder model :math:`Z = f(X; A)`, which take as input an :math:`N \times D` feature matrix X and 
    a graph adjacency matrix A, and produce an :math:`N \times E` node embedding matrix
    :math:`Z = [z_1^T,..., z_N^T ]^T`;

    2) a pairwise decoder model :math:`\hat A = g(Z)`, which takes pairs of node embeddings :math:`(z_i, z_j)` and 
    predicts respective entries :math:`\hat A_{ij}` in the adjacency matrix. 

    Note that :math:`N` denotes the number of nodes, :math:`D` the number of input features,
    and :math:`E` the embedding size.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(GCMC, self).__init__(config, dataset)

        # load dataset info
        self.num_all = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)  # csr

        # load parameters info
        self.dropout_prob = config['dropout_prob']
        self.sparse_feature = config['sparse_feature']
        self.gcn_output_dim = config['gcn_output_dim']
        self.dense_output_dim = config['embedding_size']
        self.n_class = config['class_num']
        self.num_basis_functions = config['num_basis_functions']

        # generate node feature
        if self.sparse_feature:
            features = self.get_sparse_eye_mat(self.num_all)
            i = features._indices()
            v = features._values()
            self.user_features = torch.sparse.FloatTensor(i[:, :self.n_users], v[:self.n_users],
                                                          torch.Size([self.n_users, self.num_all])).to(self.device)
            item_i = i[:, self.n_users:]
            item_i[0, :] = item_i[0, :] - self.n_users
            self.item_features = torch.sparse.FloatTensor(item_i, v[self.n_users:],
                                                          torch.Size([self.n_items, self.num_all])).to(self.device)
        else:
            features = torch.eye(self.num_all).to(self.device)
            self.user_features, self.item_features = torch.split(
                features, [self.n_users, self.n_items])
        self.input_dim = self.user_features.shape[1]

        # adj matrixs for each relation are stored in self.support
        self.Graph = self.get_norm_adj_mat().to(self.device)
        self.support = [self.Graph]

        # accumulation operation
        self.accum = config['accum']
        if self.accum == 'stack':
            div = self.gcn_output_dim // len(self.support)
            if self.gcn_output_dim % len(self.support) != 0:
                print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d (in %d splits).\n"""
                      % (self.gcn_output_dim, len(self.support) * div, len(self.support)))
            self.gcn_output_dim = len(self.support) * div

        # define layers and loss
        self.GcEncoder = GcEncoder(accum=self.accum,
                                   num_user=self.n_users,
                                   num_item=self.n_items,
                                   support=self.support,
                                   input_dim=self.input_dim,
                                   gcn_output_dim=self.gcn_output_dim,
                                   dense_output_dim=self.dense_output_dim,
                                   drop_prob=self.dropout_prob,
                                   device=self.device,
                                   sparse_feature=self.sparse_feature).to(self.device)
        self.BiDecoder = BiDecoder(input_dim=self.dense_output_dim,
                                   output_dim=self.n_class,
                                   drop_prob=0.,
                                   device=self.device,
                                   num_weights=self.num_basis_functions).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()

    def get_sparse_eye_mat(self, num):
        r"""Get the normalized sparse eye matrix.

        Construct the sparse eye matrix as node feature.

        Args:
            num: the number of rows

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        A = A.tolil()
        A[:self.n_users, self.n_users:] = self.interaction_matrix
        A[self.n_users:, :self.n_users] = self.interaction_matrix.transpose()
        A = A.todok()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, user_X, item_X, user, item):
        # Graph autoencoders are comprised of a graph encoder model and a pairwise decoder model.
        user_embedding, item_embedding = self.GcEncoder(user_X, item_X)
        predict_score = self.BiDecoder(
            user_embedding, item_embedding, user, item)
        return predict_score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        users = torch.cat((user, user))
        items = torch.cat((pos_item, neg_item))

        user_X, item_X = self.user_features, self.item_features
        predict = self.forward(user_X, item_X, users, items)
        target = torch.zeros(len(user) * 2, dtype=torch.long).to(self.device)
        target[:len(user)] = 1

        loss = self.loss_function(predict, target)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_X, item_X = self.user_features, self.item_features
        predict = self.forward(user_X, item_X, user, item)

        score = predict[:, 1]
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_X, item_X = self.user_features, self.item_features
        predict = self.forward(user_X, item_X, user, None)

        score = predict[:, 1]
        return score


class GcEncoder(nn.Module):
    """Graph Convolutional Encoder
    GcEncoder take as input an :math:`N \times D` feature matrix :math:`X` and a graph adjacency matrix :math:`A`,
    and produce an :math:`N \times E` node embedding matrix;
    Note that :math:`N` denotes the number of nodes, :math:`D` the number of input features,
    and :math:`E` the embedding size.
    """

    def __init__(self, accum, num_user, num_item, support, input_dim, gcn_output_dim, dense_output_dim, drop_prob, device,
                 sparse_feature=True, act_dense=lambda x: x, share_user_item_weights=True, bias=False):
        super(GcEncoder, self).__init__()
        self.num_users = num_user
        self.num_items = num_item
        self.input_dim = input_dim
        self.gcn_output_dim = gcn_output_dim
        self.dense_output_dim = dense_output_dim
        self.accum = accum
        self.sparse_feature = sparse_feature

        self.device = device
        self.dropout_prob = drop_prob
        self.dropout = nn.Dropout(p=self.dropout_prob)
        if self.sparse_feature:
            self.sparse_dropout = SparseDropout(p=self.dropout_prob)
        else:
            self.sparse_dropout = nn.Dropout(p=self.dropout_prob)

        self.dense_activate = act_dense
        self.activate = nn.ReLU()
        self.share_weights = share_user_item_weights
        self.bias = bias

        self.support = support
        self.num_support = len(support)

        # gcn layer
        if self.accum == 'sum':
            self.weights_u = nn.ParameterList(
                [nn.Parameter(torch.FloatTensor(self.input_dim, self.gcn_output_dim).to(self.device),
                              requires_grad=True)
                 for _ in range(self.num_support)])
            if share_user_item_weights:
                self.weights_v = self.weights_u
            else:
                self.weights_v = nn.ParameterList(
                    [nn.Parameter(torch.FloatTensor(self.input_dim, self.gcn_output_dim).to(self.device),
                                  requires_grad=True) for _ in range(self.num_support)])
        else:
            assert self.gcn_output_dim % self.num_support == 0, 'output_dim must be multiple of num_support for stackGC'
            self.sub_hidden_dim = self.gcn_output_dim // self.num_support

            self.weights_u = nn.ParameterList(
                [nn.Parameter(torch.FloatTensor(self.input_dim, self.sub_hidden_dim).to(self.device),
                              requires_grad=True)
                 for _ in range(self.num_support)])
            if share_user_item_weights:
                self.weights_v = self.weights_u
            else:
                self.weights_v = nn.ParameterList(
                    [nn.Parameter(torch.FloatTensor(self.input_dim, self.sub_hidden_dim).to(self.device),
                                  requires_grad=True) for _ in range(self.num_support)])

        # dense layer
        self.dense_layer_u = nn.Linear(
            self.gcn_output_dim, self.dense_output_dim, bias=self.bias)
        if share_user_item_weights:
            self.dense_layer_v = self.dense_layer_u
        else:
            self.dense_layer_v = nn.Linear(
                self.gcn_output_dim, self.dense_output_dim, bias=self.bias)

        self._init_weights()

    def _init_weights(self):
        init_range = math.sqrt((self.num_support + 1) /
                               (self.input_dim + self.gcn_output_dim))
        for w in range(self.num_support):
            self.weights_u[w].data.uniform_(-init_range, init_range)
        if not self.share_weights:
            for w in range(self.num_support):
                self.weights_v[w].data.uniform_(-init_range, init_range)

        dense_init_range = math.sqrt(
            (self.num_support + 1) / (self.dense_output_dim + self.gcn_output_dim))
        self.dense_layer_u.weight.data.uniform_(
            -dense_init_range, dense_init_range)
        if not self.share_weights:
            self.dense_layer_v.weight.data.uniform_(
                -dense_init_range, dense_init_range)

        if self.bias:
            self.dense_layer_u.bias.data.fill_(0)
            if not self.share_weights:
                self.dense_layer_v.bias.data.fill_(0)

    def forward(self, user_X, item_X):
        # ----------------------------------------GCN layer----------------------------------------

        user_X = self.sparse_dropout(user_X)
        item_X = self.sparse_dropout(item_X)

        embeddings = []
        if self.accum == 'sum':
            wu = 0.
            wv = 0.
            for i in range(self.num_support):
                # weight sharing
                wu = self.weights_u[i] + wu
                wv = self.weights_v[i] + wv

                # multiply feature matrices with weights
                if self.sparse_feature:
                    temp_u = torch.sparse.mm(user_X, wu)
                    temp_v = torch.sparse.mm(item_X, wv)
                else:
                    temp_u = torch.mm(user_X, wu)
                    temp_v = torch.mm(item_X, wv)
                all_embedding = torch.cat([temp_u, temp_v])

                # then multiply with adj matrices
                graph_A = self.support[i]
                all_emb = torch.sparse.mm(graph_A, all_embedding)
                embeddings.append(all_emb)

            embeddings = torch.stack(embeddings, dim=1)
            embeddings = torch.sum(embeddings, dim=1)
        else:
            for i in range(self.num_support):
                # multiply feature matrices with weights
                if self.sparse_feature:
                    temp_u = torch.sparse.mm(user_X, self.weights_u[i])
                    temp_v = torch.sparse.mm(item_X, self.weights_v[i])
                else:
                    temp_u = torch.mm(user_X, self.weights_u[i])
                    temp_v = torch.mm(item_X, self.weights_v[i])
                all_embedding = torch.cat([temp_u, temp_v])

                # then multiply with adj matrices
                graph_A = self.support[i]
                all_emb = torch.sparse.mm(graph_A, all_embedding)
                embeddings.append(all_emb)

            embeddings = torch.cat(embeddings, dim=1)

        users, items = torch.split(
            embeddings, [self.num_users, self.num_items])

        u_hidden = self.activate(users)
        v_hidden = self.activate(items)

        # ----------------------------------------Dense Layer----------------------------------------

        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)

        u_hidden = self.dense_layer_u(u_hidden)
        v_hidden = self.dense_layer_u(v_hidden)

        u_outputs = self.dense_activate(u_hidden)
        v_outputs = self.dense_activate(v_hidden)

        return u_outputs, v_outputs


class BiDecoder(nn.Module):
    """Bilinear decoder
    BiDecoder takes pairs of node embeddings and predicts respective entries in the adjacency matrix.
    """

    def __init__(self, input_dim, output_dim, drop_prob, device,
                 num_weights=3, act=lambda x: x):
        super(BiDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_weights = num_weights
        self.device = device

        self.activate = act
        self.dropout_prob = drop_prob
        self.dropout = nn.Dropout(p=self.dropout_prob)

        self.weights = nn.ParameterList(
            [nn.Parameter(orthogonal([self.input_dim, self.input_dim]).to(self.device))
             for _ in range(self.num_weights)])
        self.dense_layer = nn.Linear(
            self.num_weights, self.output_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        dense_init_range = math.sqrt(
            self.output_dim / (self.num_weights + self.output_dim))
        self.dense_layer.weight.data.uniform_(
            -dense_init_range, dense_init_range)

    def forward(self, u_inputs, i_inputs, users, items=None):
        u_inputs = self.dropout(u_inputs)
        i_inputs = self.dropout(i_inputs)

        if items is not None:
            users_emb = u_inputs[users]
            items_emb = i_inputs[items]

            basis_outputs = []
            for i in range(self.num_weights):
                users_emb_temp = torch.mm(users_emb, self.weights[i])
                scores = torch.mul(users_emb_temp, items_emb)
                scores = torch.sum(scores, dim=1)
                basis_outputs.append(scores)
        else:
            users_emb = u_inputs[users]
            items_emb = i_inputs

            basis_outputs = []
            for i in range(self.num_weights):
                users_emb_temp = torch.mm(users_emb, self.weights[i])
                scores = torch.mm(users_emb_temp, items_emb.transpose(0, 1))
                basis_outputs.append(scores.view(-1))

        basis_outputs = torch.stack(basis_outputs, dim=1)
        basis_outputs = self.dense_layer(basis_outputs)
        output = self.activate(basis_outputs)

        return output


class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) +
                 self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


def orthogonal(shape, scale=1.1):
    """
    Initialization function for weights in class GCMC.
    From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)

    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return torch.tensor(scale * q[:shape[0], :shape[1]], dtype=torch.float32)
