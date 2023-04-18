# -*- coding: utf-8 -*-
# @Time   : 2023/4/9
# @Author : Yilin Liu
# @Email  : liu_elaine2022@outlook.com
# @File   : fm2.py

r"""
FM2
#####################################################
Reference:
    Sun Y, Pan J, Zhang A, et al. “FM2: Field-matrixed Factorization Machines for Recommender Systems.” in WWW 2021.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from recbole.model.abstract_recommender import ContextRecommender


class FM2(ContextRecommender):
    """
        Field-matrixed Factorization Machines (FmFM, or 𝐹M^2 ), propose a novel approach FmFM to model the
interactions of field pairs as a matrix.
    """

    def __init__(self, config, dataset):
        super(FM2, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.num_fields = self.num_feature_field
        self.interact_dim = int(self.num_fields * (self.num_fields - 1) / 2)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.weight = torch.randn(
            self.interact_dim, self.embedding_dim, self.embedding_dim, requires_grad=True, device=self.device
        )

        # parameters initialization
        self.apply(self._init_weights)

        # used to mark cross items
        self.triu_index = torch.triu(torch.ones(self.num_fields, self.num_fields), 1).nonzero().to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def fm2_matrix(self, infeature):
        """
            FmFM interaction terms calculation
        """
        left_emb = torch.index_select(infeature, 1, self.triu_index[:, 0])  # [batch_size, interact_dim, embed_dim]
        right_emb = torch.index_select(infeature, 1, self.triu_index[:, 1])
        left_emb = torch.matmul(left_emb.unsqueeze(2), self.weight).squeeze(2)
        fm2_output = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)  # [batch_size, 1]

        return fm2_output

    def forward(self, interaction):
        fm2_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]

        output = self.first_order_linear(interaction) + self.fm2_matrix(fm2_all_embeddings)

        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
