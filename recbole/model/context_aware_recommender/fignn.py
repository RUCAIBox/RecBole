# -*- coding: utf-8 -*-
# @Time   : 2022/10/27
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

from recbole.utils import InputType
from recbole.model.abstract_recommender import ContextRecommender


class GraphLayer(nn.Module):
    """
    The implementations of the GraphLayer part and the Attentional Edge Weights part are adapted from https://github.com/xue-pai/FuxiCTR.
    """

    def __init__(self, num_fields, embedding_size):
        super(GraphLayer, self).__init__()
        self.W_in = nn.Parameter(
            torch.Tensor(num_fields, embedding_size, embedding_size)
        )
        self.W_out = nn.Parameter(
            torch.Tensor(num_fields, embedding_size, embedding_size)
        )
        xavier_normal_(self.W_in)
        xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_size))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class FiGNN(ContextRecommender):
    """FiGNN is a CTR prediction model based on GGNN,
    which can model sophisticated interactions among feature fields on the graph-structured features.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(FiGNN, self).__init__(config, dataset)

        # load parameters info
        self.attention_size = config["attention_size"]
        self.n_layers = config["n_layers"]
        self.num_heads = config["num_heads"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]

        # define layers and loss
        self.dropout_layer = nn.Dropout(p=self.hidden_dropout_prob)
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)
        # multi-head self-attention network
        self.self_attn = nn.MultiheadAttention(
            self.attention_size,
            self.num_heads,
            dropout=self.attn_dropout_prob,
            batch_first=True,
        )
        self.v_res_embedding = torch.nn.Linear(self.embedding_size, self.attention_size)
        # FiGNN
        self.src_nodes, self.dst_nodes = zip(
            *list(product(range(self.num_feature_field), repeat=2))
        )
        self.gnn = nn.ModuleList(
            [
                GraphLayer(self.num_feature_field, self.attention_size)
                for _ in range(self.n_layers - 1)
            ]
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(self.attention_size * 2, 1, bias=False)
        self.gru_cell = nn.GRUCell(self.attention_size, self.attention_size)
        # Attentional Scoring Layer
        self.mlp1 = nn.Linear(self.attention_size, 1, bias=False)
        self.mlp2 = nn.Linear(
            self.num_feature_field * self.attention_size,
            self.num_feature_field,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        # parameters initialization
        self.apply(self._init_weights)

    def fignn_layer(self, in_feature):
        emb_feature = self.att_embedding(in_feature)
        emb_feature = self.dropout_layer(emb_feature)
        # multi-head self-attention network
        att_feature, _ = self.self_attn(
            emb_feature, emb_feature, emb_feature
        )  # [batch_size, num_field, att_dim]
        # Residual connection
        v_res = self.v_res_embedding(in_feature)
        att_feature += v_res
        att_feature = F.relu(att_feature).contiguous()

        # init graph
        src_emb = att_feature[:, self.src_nodes, :]
        dst_emb = att_feature[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_feature_field, self.num_feature_field)
        mask = torch.eye(self.num_feature_field).to(self.device)
        alpha = alpha.masked_fill(mask.bool(), float("-inf"))
        self.graph = F.softmax(alpha, dim=-1)
        # message passing
        if self.n_layers > 1:
            h = att_feature
            for i in range(self.n_layers - 1):
                a = self.gnn[i](self.graph, h)
                a = a.view(-1, self.attention_size)
                h = h.view(-1, self.attention_size)
                h = self.gru_cell(a, h)
                h = h.view(-1, self.num_feature_field, self.attention_size)
                h += att_feature
        else:
            h = att_feature
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
        fignn_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        output = self.fignn_layer(fignn_all_embeddings)
        return output.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
