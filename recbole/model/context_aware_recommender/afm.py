# -*- coding: utf-8 -*-
# @Time   : 2020/7/21
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# @File   : afm.py

r"""
AFM
################################################
Reference:
    Jun Xiao et al. "Attentional Factorization Machines: Learning the Weight of Feature Interactions via
    Attention Networks" in IJCAI 2017.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import AttLayer


class AFM(ContextRecommender):
    """AFM is a attention based FM model that predict the final score with the attention of input feature."""

    def __init__(self, config, dataset):
        super(AFM, self).__init__(config, dataset)

        # load parameters info
        self.attention_size = config["attention_size"]
        self.dropout_prob = config["dropout_prob"]
        self.reg_weight = config["reg_weight"]
        self.num_pair = self.num_feature_field * (self.num_feature_field - 1) / 2

        # define layers and loss
        self.attlayer = AttLayer(self.embedding_size, self.attention_size)
        self.p = nn.Parameter(torch.randn(self.embedding_size), requires_grad=True)
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def build_cross(self, feat_emb):
        """Build the cross feature columns of feature columns

        Args:
            feat_emb (torch.FloatTensor): input feature embedding tensor. shape of [batch_size, field_size, embed_dim].

        Returns:
            tuple:
                - torch.FloatTensor: Left part of the cross feature. shape of [batch_size, num_pairs, emb_dim].
                - torch.FloatTensor: Right part of the cross feature. shape of [batch_size, num_pairs, emb_dim].
        """
        # num_pairs = num_feature_field * (num_feature_field-1) / 2
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]  # [batch_size, num_pairs, emb_dim]
        q = feat_emb[:, col]  # [batch_size, num_pairs, emb_dim]
        return p, q

    def afm_layer(self, infeature):
        """Get the attention-based feature interaction score

        Args:
            infeature (torch.FloatTensor): input feature embedding tensor. shape of [batch_size, field_size, embed_dim].

        Returns:
            torch.FloatTensor: Result of score. shape of [batch_size, 1].
        """
        p, q = self.build_cross(infeature)
        pair_wise_inter = torch.mul(p, q)  # [batch_size, num_pairs, emb_dim]

        # [batch_size, num_pairs, 1]
        att_signal = self.attlayer(pair_wise_inter).unsqueeze(dim=2)

        att_inter = torch.mul(
            att_signal, pair_wise_inter
        )  # [batch_size, num_pairs, emb_dim]
        att_pooling = torch.sum(att_inter, dim=1)  # [batch_size, emb_dim]
        att_pooling = self.dropout_layer(att_pooling)  # [batch_size, emb_dim]

        att_pooling = torch.mul(att_pooling, self.p)  # [batch_size, emb_dim]
        att_pooling = torch.sum(att_pooling, dim=1, keepdim=True)  # [batch_size, 1]

        return att_pooling

    def forward(self, interaction):
        afm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]

        output = self.first_order_linear(interaction) + self.afm_layer(
            afm_all_embeddings
        )
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        l2_loss = self.reg_weight * torch.norm(self.attlayer.w.weight, p=2)
        return self.loss(output, label) + l2_loss

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
