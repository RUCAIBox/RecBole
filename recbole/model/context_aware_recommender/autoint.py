# -*- coding: utf-8 -*-
# @Time   : 2020/09/01
# @Author : Shuqing Bian
# @Email  : shuqingbian@gmail.com
# @File   : autoint.py

r"""
AutoInt
################################################
Reference:
    Weiping Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
    in CIKM 2018.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.layers import MLPLayers
from recbole.model.abstract_recommender import ContextRecommender


class AutoInt(ContextRecommender):
    """ AutoInt is a novel CTR prediction model based on self-attention mechanism,
    which can automatically learn high-order feature interactions in an explicit fashion.

    """

    def __init__(self, config, dataset):
        super(AutoInt, self).__init__(config, dataset)

        # load parameters info
        self.attention_size = config['attention_size']
        self.dropout_probs = config['dropout_probs']
        self.n_layers = config['n_layers']
        self.num_heads = config['num_heads']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.has_residual = config['has_residual']

        # define layers and loss
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)
        self.embed_output_dim = self.num_feature_field * self.embedding_size
        self.atten_output_dim = self.num_feature_field * self.attention_size
        size_list = [self.embed_output_dim] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_probs[1])
        # multi-head self-attention network
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(self.attention_size, self.num_heads, dropout=self.dropout_probs[0])
            for _ in range(self.n_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        if self.has_residual:
            self.v_res_res_embedding = torch.nn.Linear(self.embedding_size, self.attention_size)

        self.dropout_layer = nn.Dropout(p=self.dropout_probs[2])
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def autoint_layer(self, infeature):
        """ Get the attention-based feature interaction score

        Args:
            infeature (torch.FloatTensor): input feature embedding tensor. shape of[batch_size,field_size,embed_dim].

        Returns:
            torch.FloatTensor: Result of score. shape of [batch_size,1] .
        """

        att_infeature = self.att_embedding(infeature)
        cross_term = att_infeature.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        # Residual connection
        if self.has_residual:
            v_res = self.v_res_embedding(infeature)
            cross_term += v_res
        # Interacting layer
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        batch_size = infeature.shape[0]
        att_output = self.attn_fc(cross_term) + self.deep_predict_layer(self.mlp_layers(infeature.view(batch_size, -1)))
        return att_output

    def forward(self, interaction):
        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        autoint_all_embeddings = torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]
        output = self.first_order_linear(interaction) + self.autoint_layer(autoint_all_embeddings)
        return self.sigmoid(output.squeeze(1))

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
