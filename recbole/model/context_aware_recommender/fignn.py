# -*- coding: utf-8 -*-
# @Time   : 2022/10/23
# @Author : Yuyan Zhang
# @Email  : 2019308160102@cau.edu.cn
# @File   : fignn.py

r"""
FiGNN
################################################
Reference:
    Li, Zekun, et al.  "Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction"
    in CIKM 2019.

Reference code:
    - https://github.com/CRIPAC-DIG/GraphCTR
    - https://github.com/xue-pai/FuxiCTR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from itertools import product

from recbole.model.abstract_recommender import ContextRecommender


class GraphLayer(nn.Module):

    def __init__(self, num_fields, embedding_size):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_size, embedding_size))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_size, embedding_size))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_size))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class FiGNN(ContextRecommender):
    """ FiGNN is a novel CTR prediction model based on GGNN,
    which can model sophisticated interactions among feature fields on the graph-structured features.
    """

    def __init__(self, config, dataset):
        super(FiGNN, self).__init__(config, dataset)

        # load parameters info
        self.attention_size = config['attention_size']
        self.n_layers = config['n_layers']
        self.num_heads = config['num_heads']
        self.dropout_probs = config['dropout_probs']

        # define layers and loss
        self.dropout_layer = nn.Dropout(p=self.dropout_probs[1])
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)
        # multi-head self-attention network
        self.self_attn = nn.MultiheadAttention(
            self.attention_size, self.num_heads, dropout=self.dropout_probs[0], batch_first=True
        )
        self.v_res_embedding = torch.nn.Linear(self.embedding_size, self.attention_size)
        # FiGNN
        self.src_nodes, self.dst_nodes = zip(*list(product(range(self.num_feature_field), repeat=2)))
        self.gnn = nn.ModuleList([
            GraphLayer(self.num_feature_field, self.attention_size) for _ in range(self.n_layers - 1)
        ])
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(self.attention_size * 2, 1, bias=False)
        self.gru_cell = nn.GRUCell(self.attention_size, self.attention_size)
        # Attentional Scoring Layer
        self.mlp1 = nn.Linear(self.attention_size, 1, bias=False)
        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_feature_field * self.attention_size, self.num_feature_field, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        # parameters initialization
        self.apply(self._init_weights)

    def fignn_layer(self, infeature):

        emb_infeature = self.att_embedding(infeature)
        emb_infeature = self.dropout_layer(emb_infeature)
        # multi-head self-attention network
        att_infeature, _ = self.self_attn(emb_infeature, emb_infeature, emb_infeature)
        # Residual connection
        v_res = self.v_res_embedding(infeature)
        att_infeature += v_res
        att_infeature = F.relu(att_infeature).contiguous()

        # init graph
        src_emb = att_infeature[:, self.src_nodes, :]
        dst_emb = att_infeature[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_feature_field, self.num_feature_field)
        mask = torch.eye(self.num_feature_field).to(self.device)
        alpha = alpha.masked_fill(mask.bool(), float('-inf'))
        self.graph = F.softmax(alpha, dim=-1)
        # message passing
        if self.n_layers > 1:
            h = att_infeature
            for i in range(self.n_layers - 1):
                a = self.gnn[i](self.graph, h)
            a = a.view(-1, self.attention_size)
            h = h.view(-1, self.attention_size)
            h = self.gru_cell(a, h)
            h = h.view(-1, self.num_feature_field, self.attention_size)
            h += att_infeature
        else:
            h = att_infeature
        # Attentional Scoring Layer
        score = self.mlp1(h).squeeze(-1)
        weight = self.mlp2(h.flatten(start_dim=1))
        logit = (weight * score).sum(dim=1).unsqueeze(-1)
        return logit

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, interaction):
        fignn_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        output = self.fignn_layer(fignn_all_embeddings)
        return self.sigmoid(output.squeeze(1))

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
