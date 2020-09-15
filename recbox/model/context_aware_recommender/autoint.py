# -*- coding: utf-8 -*-
# @Time   : 2020/09/01 15:30
# @Author : Shuqing Bian
# @Email  : shuqingbian@gmail.com
# @File   : autoint.py

"""
Reference:
"AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks" in CIKM 2018.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbox.model.layers import AttLayer, MLPLayers
from recbox.model.context_aware_recommender.context_recommender import ContextRecommender


class AUTOINT(ContextRecommender):

    def __init__(self, config, dataset):
        super(AUTOINT, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']

        self.attention_size = config['attention_size']
        self.dropout = config['dropout']
        self.num_layers = config['num_layers']
        self.weight_decay = config['weight_decay']
        self.num_heads = config['num_heads']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.has_residual = config['has_residual']
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)

        self.embed_output_dim = self.num_feature_field * self.embedding_size
        self.atten_output_dim = self.num_feature_field * self.attention_size


        size_list = [self.embed_output_dim] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout[1])
        self.self_attns = nn.ModuleList([
                nn.MultiheadAttention(self.attention_size, self.num_heads, dropout=self.dropout[0]) for _ in range(self.num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        if self.has_residual:
            self.v_res_res_embedding = torch.nn.Linear(self.embedding_size, self.attention_size)

        self.dropout_layer = nn.Dropout(p=self.dropout[2])
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def autoint_layer(self, infeature):
        """
        Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

        Output shape
        - 3D tensor with shape: ``(batch_size,1)`` .
        """

        att_infeature = self.att_embedding(infeature)
        cross_term = att_infeature.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            v_res = self.v_res_embedding(infeature)
            cross_term += v_res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        batch_size = infeature.shape[0]
        x = self.attn_fc(cross_term) + self.deep_predict_layer(self.mlp_layers(infeature.view(batch_size, -1)))
        return x


    def forward(self, interaction):
        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        x = []
        if sparse_embedding is not None:
            x.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            x.append(dense_embedding)
        x = torch.cat(x, dim=1)  # [batch_size, num_field, embed_dim]
        x = self.first_order_linear(interaction) + self.autoint_layer(x)
        return self.sigmoid(x.squeeze(1))


    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
