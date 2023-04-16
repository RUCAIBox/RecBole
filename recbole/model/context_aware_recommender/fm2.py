# -*- coding: utf-8 -*-
# @Time   : 2023/4/9
# @Author : Yilin Liu
# @Email  : liu_elaine2022@outlook.com
# @File   : fm2.py

r"""
FmFM
#####################################################
Reference:
    Sun Y, Pan J, Zhang A, et al. “FM2: Field-matrixed Factorization Machines for Recommender Systems.” in WWW 2021.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from recbole.model.abstract_recommender import ContextRecommender

class FM2(ContextRecommender):
    """"""
    def __init__(self, config, dataset):
        super(FM2, self).__init__(config, dataset)

        # self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.embedding_dim = config["embedding_dim"]

        self.num_fields = self.num_feature_field

        self.interact_dim = int(self.num_fields * (self.num_fields - 1) / 2)  #


        # self.fields = config["fields"]
        # self.num_features = self.num_feature_field

        # self.embedding_dim = config["embedding_dim"]

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.weight = torch.randn(
            self.interact_dim, self.embedding_dim, self.embedding_dim, requires_grad=True, device=self.device
        )
        # parameters initialization
        self.apply(self._init_weights)

        self.triu_index = torch.triu(torch.ones(self.num_fields, self.num_fields), 1).nonzero().to(self.device)

        # self.feature_names = (
        #     self.token_field_names,
        #     self.token_seq_field_names,
        #     self.float_field_names,
        # )
        # self.feat_dims = (
        #     self.token_field_dims,
        #     self.token_seq_field_dims,
        #     self.float_field_dims,
        # )
        # self.interaction_weight = nn.Parameter(torch.Tensor(self.interact_dim, self.embedding_dim, self.embedding_dim))




    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
           xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def fm2_matrix(self, infeature):

        # batch_size = infeature.shape[0]
        # fm2_inter = []
        # for i, feature_emb in enumerate(infeature):
        #
        #     left_emb = torch.index_select(feature_emb, 1, self.triu_index[:, 0])
        #     right_emb = torch.index_select(feature_emb, 1, self.triu_index[:, 1])
        #     left_emb = torch.matmul(left_emb.unsqueeze(2), self.weight).squeeze(2)
        #     fm2_in = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)
        #     fm2_inter.append(fm2_in)

        feature_emb = infeature
        left_emb = torch.index_select(feature_emb, 1, self.triu_index[:, 0])
        right_emb = torch.index_select(feature_emb, 1, self.triu_index[:, 1])
        left_emb = torch.matmul(left_emb, self.weight).squeeze(2)
        fm2_output = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)

        # # Reduce to [N, 1]
        # fm2_output = torch.sum(fm2_inter, dim=1, keepdim=True)  # [batch_size, 1]

        # weight = self.weight.expand(batch_size, -1, -1, -1)
        # print(infeature)




        # l_left, l_right = [], []
        # field_order = sorted(self.feature_dims.items(), key=lambda x: (-x[1], x[0]))
        # for idx_l, (feat_l, dim_l) in enumerate(field_order):
        #     for idx_r, (feat_r, dim_r) in enumerate(field_order[idx_l + 1:]):
        #         idx_r += (idx_l + 1)
        #         name = '%s_%s' % (feat_l, feat_r)
        #         name_alt = '%s_%s' % (feat_r, feat_l)
        #         if self.cross_fields and name not in self.cross_fields and name_alt not in self.cross_fields:
        #             continue
        #
        #         l_left.append(torch.matmul(self.xv[feat_l], w))
        #         l_right.append(self.xv[feat_r])
        #
        #     l_left = torch.cat(l_left, 1)
        #     l_right = torch.cat(l_right, 1)
        #     fm2_inter = torch.mul(l_left, l_right)
        #
        #
        #     # Reduce to [N, 1]
        #     fm2_output = torch.sum(fm2_inter, dim=1, keepdim=True)  # [batch_size, 1]
        #
        #     batch_size = infeature.shape[0]
        #     weight = self.weight.expand(batch_size, -1, -1, -1)
        #
        #     fwfm_inter = list()  # [batch_size, num_fields, emb_dim]
        #     for i in range(self.num_features - 1):
        #         for j in range(i + 1, self.num_features):
        #             Fi, Fj = self.feature2field[i], self.feature2field[j]
        #             fwfm_inter.append(infeature[:, i] * infeature[:, j] * weight[:, Fi, Fj])
        #     fwfm_inter = torch.stack(fwfm_inter, dim=1)
        #     fwfm_inter = torch.sum(fwfm_inter, dim=1)  # [batch_size, emb_dim]

        return fm2_output

    def forward(self, interaction):
        fm2_all_embeddings = self.concat_embed_input_fields(
           interaction
        )  # [batch_size, num_field, embed_dim]

        output = self.first_order_linear(interaction) + self.fm2_matrix(
           fm2_all_embeddings
        )

        return output.squeeze(-1)


    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))