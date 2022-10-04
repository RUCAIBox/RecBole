# -*- coding: utf-8 -*-
# @Time   : 2020/10/06
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn
# @File   : fwfm.py

r"""
FwFM
#####################################################
Reference:
    Junwei Pan et al. "Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising."
    in WWW 2018.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender


class FwFM(ContextRecommender):
    r"""FwFM is a context-based recommendation model. It aims to model the different feature interactions
    between different fields in a much more memory-efficient way. It proposes a field pair weight matrix
    :math:`r_{F(i),F(j)}`, to capture the heterogeneity of field pair interactions.

    The model defines as follows:

    .. math::
       y = w_0 + \sum_{i=1}^{m}x_{i}w_{i} + \sum_{i=1}^{m}\sum_{j=i+1}^{m}x_{i}x_{j}<v_{i}, v_{j}>r_{F(i),F(j)}
    """

    def __init__(self, config, dataset):
        super(FwFM, self).__init__(config, dataset)

        # load parameters info
        self.dropout_prob = config["dropout_prob"]
        self.fields = config["fields"]  # a dict; key: field_id; value: feature_list

        self.num_features = self.num_feature_field

        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.sigmoid = nn.Sigmoid()

        self.feature2id = {}
        self.feature2field = {}

        self.feature_names = (
            self.token_field_names,
            self.token_seq_field_names,
            self.float_field_names,
        )
        self.feature_dims = (
            self.token_field_dims,
            self.token_seq_field_dims,
            self.float_field_dims,
        )
        self._get_feature2field()
        self.num_fields = len(set(self.feature2field.values()))  # the number of fields
        self.num_pair = self.num_fields * self.num_fields
        self.weight = torch.randn(
            self.num_fields, self.num_fields, 1, requires_grad=True, device=self.device
        )
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

    def _get_feature2field(self):
        r"""Create a mapping between features and fields."""
        fea_id = 0
        for names in self.feature_names:
            if names is not None:
                for name in names:
                    self.feature2id[name] = fea_id
                    fea_id += 1

        if self.fields is None:
            field_id = 0
            for key, value in self.feature2id.items():
                self.feature2field[self.feature2id[key]] = field_id
                field_id += 1
        else:
            for key, value in self.fields.items():
                for v in value:
                    try:
                        self.feature2field[self.feature2id[v]] = key
                    except:
                        pass

    def fwfm_layer(self, infeature):
        r"""Get the field pair weight matrix r_{F(i),F(j)}, and model the different interaction strengths of
        different field pairs :math:`\sum_{i=1}^{m}\sum_{j=i+1}^{m}x_{i}x_{j}<v_{i}, v_{j}>r_{F(i),F(j)}`.

        Args:
            infeature (torch.cuda.FloatTensor): [batch_size, field_size, embed_dim]

        Returns:
            torch.cuda.FloatTensor: [batch_size, 1]
        """
        # get r(Fi, Fj)
        batch_size = infeature.shape[0]
        weight = self.weight.expand(batch_size, -1, -1, -1)

        fwfm_inter = list()  # [batch_size, num_fields, emb_dim]
        for i in range(self.num_features - 1):
            for j in range(i + 1, self.num_features):
                Fi, Fj = self.feature2field[i], self.feature2field[j]
                fwfm_inter.append(infeature[:, i] * infeature[:, j] * weight[:, Fi, Fj])
        fwfm_inter = torch.stack(fwfm_inter, dim=1)
        fwfm_inter = torch.sum(fwfm_inter, dim=1)  # [batch_size, emb_dim]
        fwfm_inter = self.dropout_layer(fwfm_inter)

        fwfm_output = torch.sum(fwfm_inter, dim=1, keepdim=True)  # [batch_size, 1]

        return fwfm_output

    def forward(self, interaction):
        fwfm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]

        output = self.first_order_linear(interaction) + self.fwfm_layer(
            fwfm_all_embeddings
        )

        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
