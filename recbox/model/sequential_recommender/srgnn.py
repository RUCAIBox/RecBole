# -*- coding: utf-8 -*-
# @Time   : 2020/9/30 14:07
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com
r"""
recbox.model.sequential_recommender.srgnn
################################################

Reference:
Shu Wu et al. "Session-based Recommendation with Graph Neural Networks." in AAAI 2019.

Reference code:
https://github.com/CRIPAC-DIG/SR-GNN

"""
import torch
import numpy as np
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import math
from recbox.utils import InputType
from recbox.model.abstract_recommender import SequentialRecommender


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    Gated gnn is a neural unit similar to gru.
    """
    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size,
                                           self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(self.embedding_size,
                                        self.embedding_size,
                                        bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size,
                                         self.embedding_size,
                                         bias=True)

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of [batch_size, max_session_len, embedding_size]

        Returns:hy(torch.FloatTensor):Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = torch.matmul(A[:, :, :A.size(1)],
                                self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)],
                                 self.linear_edge_out(hidden)) + self.b_ioh
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embdding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = (1 - inputgate) * hidden + inputgate * newgate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNN(SequentialRecommender):
    r"""SRGNN regards the conversation history as a directed graph.
    In addition to considering the connection between the item and the adjacent item,
    it also considers the connection with other interactive items.

    Such as:A example of a session sequence and the connecion matrix A
    session sequence :item1, item2, item3, item2, item4

    Outgoing edges:
         1     2    3   4
        ===== ===== ===== =====
    1     0    1     0    0
    2     0    0    1/2  1/2
    3     0    1     0    0
    4     0    0     0    0
        ===== ===== ===== =====

    Incoming edges:
         1     2    3   4
        ===== ===== ===== =====
    1     0    0      0     0
    2    1/2   0    1/2     0
    3     0    1      0     0
    4     0    1      0     0
        ===== ===== ===== =====
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SRGNN, self).__init__()
        # load parameters info
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.embedding_size = config['embedding_size']
        self.step = config['step']
        self.device = config['device']
        self.item_count = dataset.item_num
        # item embedding
        self.item_list_embedding = nn.Embedding(self.item_count,
                                                self.embedding_size,
                                                padding_idx=0)
        # define layers and loss
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_one = nn.Linear(self.embedding_size,
                                    self.embedding_size,
                                    bias=True)
        self.linear_two = nn.Linear(self.embedding_size,
                                    self.embedding_size,
                                    bias=True)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2,
                                          self.embedding_size,
                                          bias=True)
        self.criterion = nn.CrossEntropyLoss()
        # parameters init
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_slice(self, interaction):
        r"""Get the input needed by the graph neural network

        Returns:
            alias_inputs(torch.LongTensor):The relative coordinates of the item node, shape of [batch_size, max_session_len]
            A(torch.FloatTensor):The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
            items(torch.LongTensor):The unique item nodes, shape of [batch_size, max_session_len]
            mask(torch.LongTensor):Mask matrix, shape of [batch_size, max_session_len]

        """
        item_id_list = interaction[self.ITEM_ID_LIST]
        mask = item_id_list.gt(0)

        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_id_list.size(1)

        item_id_list = item_id_list.cpu().numpy()

        for u_input in item_id_list:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(A).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    def get_item_lookup_table(self):
        r"""Get the transpose of item_list_embedding.weight，Shape of (embedding_size, item_count+padding_id)
        Used to calculate the score for each item with the predict_behavior_emb
        """
        return self.item_list_embedding.weight.t()

    def forward(self, interaction):
        alias_inputs, A, items, mask = self.get_slice(interaction)
        item_list_len = interaction[self.ITEM_LIST_LEN]
        hidden = self.item_list_embedding(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1),
                                         1).expand(-1, -1, self.embedding_size)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_list_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(
            alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        predict_emb = self.linear_transform(torch.cat([a, ht], dim=1))
        return predict_emb

    def calculate_loss(self, interaction):
        target_id = interaction[self.TARGET_ITEM_ID]
        pred = self.forward(interaction)
        logits = torch.matmul(pred, self.get_item_lookup_table())
        loss = self.criterion(logits, target_id)
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        pred = self.forward(interaction)
        scores = torch.matmul(pred, self.get_item_lookup_table())
        return scores
