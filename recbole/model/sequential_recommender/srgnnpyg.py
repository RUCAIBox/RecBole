# @Time   : 2021/7/15
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
SRGNNPyG
################################################

Reference:
    Shu Wu et al. "Session-based Recommendation with Graph Neural Networks." in AAAI 2019.
    Implemented using PyTorch Geometric.

Reference code:
    https://github.com/CRIPAC-DIG/SR-GNN

"""
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        super(SRGNNConv, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(dim, dim)
        self.b = nn.Parameter(torch.Tensor(dim))

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x) + self.b


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

    def forward(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy


class SRGNNPyG(SequentialRecommender):
    r"""SRGNN regards the conversation history as a directed graph.
    In addition to considering the connection between the item and the adjacent item,
    it also considers the connection with other interactive items.

    Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connection matrix A

    Outgoing edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     1     0     0
         2    0     0    1/2   1/2
         3    0     1     0     0
         4    0     0     0     0
        === ===== ===== ===== =====

    Incoming edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     0     0     0
         2   1/2    0    1/2    0
         3    0     1     0     0
         4    0     1     0     0
        === ===== ===== ===== =====
    """

    def __init__(self, config, dataset):
        super(SRGNNPyG, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.step = config['step']
        self.device = config['device']
        self.loss_type = config['loss_type']

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        # define layers and loss
        self.gnncell = SRGNNCell(self.embedding_size)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def graph_construction(self, item_seq, item_seq_len):
        x = [torch.zeros([1], dtype=torch.long, device=item_seq.device)]
        edge_index = []
        alias_inputs = []

        tot_node_num = torch.ones([1], device=item_seq.device, dtype=torch.long)
        for i, seq in enumerate(list(torch.chunk(item_seq, item_seq.shape[0]))):
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0) + tot_node_num
            alias_seq[item_seq_len[i]:] = 0
            alias_inputs.append(alias_seq)
            short_seq = alias_seq[:item_seq_len[i]]
            # No repeat click
            edge = torch.stack([short_seq[:-1], short_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)
            tot_node_num += seq.shape[0]

        x = torch.cat(x)
        edge_index = torch.cat(edge_index, dim=-1)
        alias_inputs = torch.stack(alias_inputs)
        return x, edge_index, alias_inputs

    def forward(self, item_seq, item_seq_len):
        mask = item_seq.gt(0)
        x, edge_index, alias_inputs = self.graph_construction(item_seq, item_seq_len)
        x = self.item_embedding(x)
        hidden = x
        for i in range(self.step):
            hidden = self.gnncell(hidden, edge_index)

        seq_hidden = hidden[alias_inputs]
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
