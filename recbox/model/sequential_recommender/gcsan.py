# -*- coding: utf-8 -*-
# @Time   : 2020/10/4 16:55
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com


r"""
recbox.model.sequential_recommender.gcsan
################################################

Reference:
Chengfeng Xu et al. "Graph Contextualized Self-Attention Network for Session-based Recommendation." in IJCAI 2019.

"""
import torch
import numpy as np
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import math
from torch.nn.init import xavier_normal_, constant_
from recbox.model.loss import EmbLoss
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
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        # parameters initialization
        self.reset_parameters()



    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of [batch_size, max_session_len, embedding_size]

        Returns:
            hy(torch.FloatTensor):Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden))
        input_out = torch.matmul(A[:, :, A.size(1): 2 * A.size(1)], self.linear_edge_out(hidden))
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


class SelfAttention(nn.Module):
    r"""Self-Attention is a special case of the attention mechanism, it can draw global dependencies
    between input and output, and capture item-item transitions across the entire input and output
    sequence itself without regard to their distances.

    Self-Attention mechanism are comprised of:

    1) self-attention layer :math:`F = softmax(\frac{(HW^Q)(HW^k)^T}{\sqrt{d}})(HW^V)`,
    which takes the latent vectors of all noeds involved in the session graph as an input, such as
    :math:`H = [h_1, h_2,...,h_n]`

    2) point-wise feed-forward network :math:`E = ReLU(FW_1 + b_1)W_2+b_2+F`, which takes the result of
    the self-attention layer :math:`F` as an input

    """
    def __init__(self, embedding_size, hidden_size, dropout):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.W_Q = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W_K = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W_V = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        self.fc1 = nn.Linear(self.embedding_size, self.hidden_size, bias=True)
        self.drop1 = nn.Dropout(self.dropout[0])
        self.fc2 = nn.Linear(self.hidden_size, self.embedding_size, bias=True)
        self.drop2 = nn.Dropout(self.dropout[1])
        self.relu = nn.ReLU()

    def scale_dot_product_attention(self, Q, K, V, mask=None):
        r"""This is a function that count the attention weights

        Args:
           Q(torch.FloatTensor): query, shape of [batch_size, time_steps, emb]
           K(torch.FloatTensor): key, shape of [batch_size, time_steps, emb]
           V(torch.FloatTensor): value, shape of [batch_size, time_steps, emb]
           mask(torch.LongTensor): Avoid the influence of padding position on attention weights, shape of [batch_size, time_steps, time_steps]

        Returns:
           torch.Tensor:context, shape of [batch_size, time_steps, emb]
           torch.Tensor:attn, shape of [batch_size, time_steps, time_steps]
        """
        d = Q.size(2)
        scores = torch.matmul(Q, K.permute(0, 2, 1)) / torch.sqrt(torch.tensor(d, dtype=torch.float))
        if mask is not None:
            scores.masked_fill_(mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn



    def feedforward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x + residual
        return x

    def forward(self, encoder_outputs, mask=None):
        Q = self.W_Q(encoder_outputs)
        K = self.W_K(encoder_outputs)
        V = self.W_V(encoder_outputs)

        context, _ = self.scale_dot_product_attention(Q, K, V, mask)
        outputs = self.feedforward(context)
        return outputs


class GCSAN(SequentialRecommender):
    r"""GCSAN captures rich local dependencies via graph nerual network,
     and learns long-range dependencies by applying the self-attention mechanism.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(GCSAN, self).__init__()
        # load parameters info
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.step = config['step']
        self.num_blocks = config['num_blocks']
        self.device = config['device']
        self.dropout = config['dropout']
        self.weight = config['weight']
        self.reg_weight = config['reg_weight']
        self.item_count = dataset.item_num
        # item embedding
        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)
        # define layers and loss
        self.gnn = GNN(self.embedding_size, self.step)
        self.self_attention = SelfAttention(self.embedding_size, self.hidden_size, self.dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()
        # parameters initialization
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

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


    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)

    def forward(self, interaction):
        assert self.weight >= 0 and self.weight <= 1
        alias_inputs, A, items, mask = self.get_slice(interaction)
        item_list_len = interaction[self.ITEM_LIST_LEN]
        hidden = self.item_list_embedding(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.embedding_size)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_list_len - 1)
        a = seq_hidden
        padding_mask = self.get_attn_pad_mask(mask, mask)
        for i in range(self.num_blocks):
            a = self.self_attention(a, padding_mask)
        at = self.gather_indexes(a, item_list_len - 1)
        predict_emb = self.weight * at + (1 - self.weight) * ht
        return predict_emb

    def calculate_loss(self, interaction):
        target_id = interaction[self.TARGET_ITEM_ID]
        pred = self.forward(interaction)
        logits = torch.matmul(pred, self.get_item_lookup_table())
        loss = self.criterion(logits, target_id)
        reg_loss = self.reg_loss(self.item_list_embedding.weight)
        total_loss = loss + reg_loss
        return total_loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        pred = self.forward(interaction)
        scores = torch.matmul(pred, self.get_item_lookup_table())
        return scores
