# -*- coding: utf-8 -*-
# @Time   : 2020/10/04
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn
# @File   : ffm.py

r"""
FFM
#####################################################
Reference:
    Yuchin Juan et al. "Field-aware Factorization Machines for CTR Prediction" in RecSys 2016.

Reference code:
    https://github.com/rixwew/pytorch-fm
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender


class FFM(ContextRecommender):
    r"""FFM is a context-based recommendation model. It aims to model the different feature interactions 
    between different fields. Each feature has several latent vectors :math:`v_{i,F(j)}`,
    which depend on the field of other features, and one of them is used to do the inner product.

    The model defines as follows:

    .. math::
       y = w_0 + \sum_{i=1}^{m}x_{i}w_{i} + \sum_{i=1}^{m}\sum_{j=i+1}^{m}x_{i}x_{j}<v_{i,F(j)}, v_{j,F(i)}>
    """

    def __init__(self, config, dataset):
        super(FFM, self).__init__(config, dataset)

        # load parameters info
        self.fields = config['fields'] # a dict; key: field_id; value: feature_list

        self.sigmoid = nn.Sigmoid()

        self.feature2id = {}
        self.feature2field = {}
        
        self.feature_names = (self.token_field_names, self.float_field_names, self.token_seq_field_names)
        self.feature_dims = (self.token_field_dims, self.float_field_dims, self.token_seq_field_dims)
        self._get_feature2field()
        self.num_fields = len(set(self.feature2field.values()))  # the number of fields

        self.ffm = FieldAwareFactorizationMachine(self.feature_names, self.feature_dims, self.feature2id, self.feature2field, self.num_fields, self.embedding_size, self.device)
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

    def _get_feature2field(self):
        r"""Create a mapping between features and fields.

        """
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

    def get_ffm_input(self, interaction):
        r"""Get different types of ffm layer's input.

        """
        token_ffm_input = []
        if self.token_field_names is not None:
            for tn in self.token_field_names:
                token_ffm_input.append(torch.unsqueeze(interaction[tn], 1))
            if len(token_ffm_input) > 0:
                token_ffm_input = torch.cat(token_ffm_input, dim=1) # [batch_size, num_token_features]
        float_ffm_input = []
        if self.float_field_names is not None:
            for fn in self.float_field_names:
                float_ffm_input.append(torch.unsqueeze(interaction[fn], 1))
            if len(float_ffm_input) > 0:
                float_ffm_input = torch.cat(float_ffm_input, dim=1) # [batch_size, num_float_features]
        token_seq_ffm_input = []
        if self.token_seq_field_names is not None:
            for tsn in self.token_seq_field_names:
                token_seq_ffm_input.append(interaction[tsn]) # a list

        return (token_ffm_input, float_ffm_input, token_seq_ffm_input)

    def forward(self, interaction):
        ffm_input = self.get_ffm_input(interaction)
        ffm_output = torch.sum(torch.sum(self.ffm(ffm_input), dim=1), dim=1, keepdim=True)
        output = self.sigmoid(self.first_order_linear(interaction) + ffm_output)
        
        return output.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)


class FieldAwareFactorizationMachine(nn.Module):
    r"""This is Field-Aware Factorization Machine Module for FFM.

    """

    def __init__(self, feature_names, feature_dims, feature2id, feature2field, num_fields, embed_dim, device):
        super(FieldAwareFactorizationMachine, self).__init__()

        self.token_feature_names = feature_names[0]
        self.float_feature_names = feature_names[1]
        self.token_seq_feature_names = feature_names[2]
        self.token_feature_dims = feature_dims[0]
        self.float_feature_dims = feature_dims[1]
        self.token_seq_feature_dims = feature_dims[2]

        self.feature2id = feature2id
        self.feature2field = feature2field
        self.num_features = len(self.token_feature_names) + len(self.float_feature_names) + len(self.token_seq_feature_names)
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.device = device

        # init token field-aware embeddings if there is token type of features.
        if len(self.token_feature_names) > 0:
            self.num_token_features = len(self.token_feature_names)
            self.token_embeddings = torch.nn.ModuleList([
                nn.Embedding(sum(self.token_feature_dims), self.embed_dim) for _ in range(self.num_fields)
            ])
            self.token_offsets = np.array((0, *np.cumsum(self.token_feature_dims)[:-1]), dtype=np.long)
            for embedding in self.token_embeddings:
                nn.init.xavier_uniform_(embedding.weight.data)
        # init float field-aware embeddings if there is float type of features.
        if len(self.float_feature_names) > 0:
            self.num_float_features = len(self.float_feature_names)
            self.float_embeddings = nn.Embedding(np.sum(self.token_feature_dims, dtype=np.int32), self.embed_dim)
            self.float_embeddings = torch.nn.ModuleList([
                nn.Embedding(self.num_float_features, self.embed_dim) for _ in range(self.num_fields)
            ])
            for embedding in self.float_embeddings:
                nn.init.xavier_uniform_(embedding.weight.data)
        # init token_seq field-aware embeddings if there is token_seq type of features.
        if len(self.token_seq_feature_names) > 0:
            self.num_token_seq_features = len(self.token_seq_feature_names)
            self.token_seq_embeddings = torch.nn.ModuleList()
            self.token_seq_embedding = torch.nn.ModuleList()
            for i in range(self.num_fields):
                for token_seq_feature_dim in self.token_seq_feature_dims:
                    self.token_seq_embedding.append(nn.Embedding(token_seq_feature_dim, self.embed_dim))
                for embedding in self.token_seq_embedding:
                    nn.init.xavier_uniform_(embedding.weight.data)
                self.token_seq_embeddings.append(self.token_seq_embedding)

    def forward(self, input_x):
        r"""Model the different interaction strengths of different field pairs.
        

        Args:
            input_x (a tuple): (token_ffm_input, float_ffm_input, token_seq_ffm_input)

                    token_ffm_input (torch.cuda.FloatTensor): [batch_size, num_token_features] or None

                    float_ffm_input (torch.cuda.FloatTensor): [batch_size, num_float_features] or None

                    token_seq_ffm_input (list): length is num_token_seq_features or 0

        Returns:
            torch.cuda.FloatTensor: The results of all features' field-aware interactions.
            shape: [batch_size, num_fields, emb_dim]
        """
        token_ffm_input, float_ffm_input, token_seq_ffm_input = input_x[0], input_x[1], input_x[2]

        token_input_x_emb = self._emb_token_ffm_input(token_ffm_input)
        float_input_x_emb = self._emb_float_ffm_input(float_ffm_input)
        token_seq_input_x_emb = self._emb_token_seq_ffm_input(token_seq_ffm_input)
        
        input_x_emb = self._get_input_x_emb(token_input_x_emb, float_input_x_emb, token_seq_input_x_emb)

        output = list()
        for i in range(self.num_features - 1):
            for j in range(i + 1, self.num_features):
                output.append(input_x_emb[self.feature2field[j]][:, i] * input_x_emb[self.feature2field[i]][:, j])
        output = torch.stack(output, dim=1)  # [batch_size, num_fields, emb_dim]

        return output

    def _get_input_x_emb(self, token_input_x_emb, float_input_x_emb, token_seq_input_x_emb):
        # merge different types of field-aware embeddings
        input_x_emb = []  # [num_fields: [batch_size, num_fields, emb_dim]]
        if len(self.token_feature_names) > 0 and len(self.float_feature_names) > 0 and len(self.token_seq_feature_names) > 0:
            for i in range(self.num_fields):
                input_x_emb.append(torch.cat([token_input_x_emb[i], float_input_x_emb[i], token_seq_input_x_emb[i]], dim=1))
        elif len(self.token_feature_names) > 0 and len(self.float_feature_names) > 0:
            for i in range(self.num_fields):
                input_x_emb.append(torch.cat([token_input_x_emb[i], float_input_x_emb[i]], dim=1))
        elif len(self.float_feature_names) > 0 and len(self.token_seq_feature_names) > 0:
            for i in range(self.num_fields):
                input_x_emb.append(torch.cat([float_input_x_emb[i], token_seq_input_x_emb[i]], dim=1))
        elif len(self.token_feature_names) > 0 and len(self.token_seq_feature_names) > 0:
            for i in range(self.num_fields):
                input_x_emb.append(torch.cat([token_input_x_emb[i], token_seq_input_x_emb[i]], dim=1))
        elif len(self.token_feature_names) > 0:
            input_x_emb = token_input_x_emb
        elif len(self.float_feature_names) > 0:
            input_x_emb = float_input_x_emb
        elif len(self.token_seq_feature_names) > 0:
            input_x_emb = token_seq_input_x_emb

        return input_x_emb

    def _emb_token_ffm_input(self, token_ffm_input):
        # get token field-aware embeddings
        token_input_x_emb = []
        if len(self.token_feature_names) > 0:
            token_input_x = token_ffm_input + token_ffm_input.new_tensor(self.token_offsets).unsqueeze(0)
            token_input_x_emb = [self.token_embeddings[i](token_input_x) for i in range(self.num_fields)] # [num_fields: [batch_size, num_token_features, emb_dim]]

        return token_input_x_emb

    def _emb_float_ffm_input(self, float_ffm_input):
        # get float field-aware embeddings
        float_input_x_emb = []
        if len(self.float_feature_names) > 0:
            index = torch.arange(0, self.num_float_features).unsqueeze(0).expand_as(float_ffm_input).long().to(self.device) # [batch_size, num_float_features]
            float_input_x_emb = [torch.mul(self.float_embeddings[i](index), float_ffm_input.unsqueeze(2)) for i in range(self.num_fields)] # [num_fields: [batch_size, num_float_features, emb_dim]]

        return float_input_x_emb

    def _emb_token_seq_ffm_input(self, token_seq_ffm_input):
        # get token_seq field-aware embeddings
        token_seq_input_x_emb = []
        if len(self.token_seq_feature_names) > 0:
            for i in range(self.num_fields):
                token_seq_result = []
                for j, token_seq in enumerate(token_seq_ffm_input):
                    embedding_table = self.token_seq_embeddings[i][j]
                    mask = token_seq != 0  # [batch_size, seq_len]
                    mask = mask.float()
                    value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

                    token_seq_embedding = embedding_table(token_seq)  # [batch_size, seq_len, embed_dim]
                    mask = mask.unsqueeze(2).expand_as(token_seq_embedding)  # [batch_size, seq_len, embed_dim]
                    # mean
                    masked_token_seq_embedding = token_seq_embedding * mask.float()
                    result = torch.sum(masked_token_seq_embedding, dim=1)  # [batch_size, embed_dim]
                    eps = torch.FloatTensor([1e-8]).to(self.device)
                    result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                    result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]

                    token_seq_result.append(result)
                token_seq_input_x_emb.append(torch.cat(token_seq_result, dim=1)) # [num_fields: batch_size, num_token_seq_features, embed_dim]

        return token_seq_input_x_emb
