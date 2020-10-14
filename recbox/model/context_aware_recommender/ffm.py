# -*- coding: utf-8 -*-
# @Time   : 2020/10/04
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn
# @File   : ffm.py

r"""
recbox.model.context_aware_recommender.ffm
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
from recbox.model.context_aware_recommender.context_recommender import ContextRecommender


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

        self.LABEL = config['LABEL_FIELD']
        self.sigmoid = nn.Sigmoid()
        self.all_field_dims = (self.token_field_dims, self.float_field_dims)
        self.ffm = FieldAwareFactorizationMachine(self.all_field_dims, self.embedding_size, self.device)
        self.loss = nn.BCELoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_ffm_input(self, interaction):
        token_ffm_input = []
        if self.token_field_names is not None:
            for tn in self.token_field_names:
                token_ffm_input.append(torch.unsqueeze(interaction[tn], 1))
            if len(token_ffm_input) > 0:
                token_ffm_input = torch.cat(token_ffm_input, dim=1) # [batch_size, num_token_fields]
        float_ffm_input = []
        if self.float_field_names is not None:
            for fn in self.float_field_names:
                float_ffm_input.append(torch.unsqueeze(interaction[fn], 1))
            if len(float_ffm_input) > 0:
                float_ffm_input = torch.cat(float_ffm_input, dim=1) # [batch_size, num_float_fields]

        return token_ffm_input, float_ffm_input

    def forward(self, interaction):
        # ffm_input: [batch_size, num_token_fields] or [batch_size, num_float_fields]
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

    def __init__(self, all_field_dims, embed_dim, device):
        super(FieldAwareFactorizationMachine, self).__init__()
        self.token_field_dims = all_field_dims[0]
        self.float_field_dims = all_field_dims[1]
        self.field_dims = self.token_field_dims + self.float_field_dims
        self.device = device
        # init token field-aware embeddings
        if self.token_field_dims is not None:
            self.token_num_fields = len(self.token_field_dims)
            self.token_embeddings = torch.nn.ModuleList([
                nn.Embedding(sum(self.token_field_dims), embed_dim) for _ in range(len(self.field_dims))
            ])
            self.token_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            for embedding in self.token_embeddings:
                nn.init.xavier_uniform_(embedding.weight.data)
        # init float field-aware embeddings
        if self.float_field_dims is not None:
            self.float_num_fields = len(self.float_field_dims)
            self.float_embeddings = nn.Embedding(np.sum(self.float_field_dims, dtype=np.int32),
                                                      embed_dim)
            self.float_embeddings = torch.nn.ModuleList([
                nn.Embedding(self.float_num_fields, embed_dim) for _ in range(len(self.field_dims))
            ])
            for embedding in self.float_embeddings:
                nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, input_x):
        r"""
        Args:
            input_x (torch.cuda.FloatTensor): ``(batch_size, num_token_fields)`` or ``(batch_size, num_token_fields)``

        Returns:
            torch.cuda.FloatTensor: The result of all features' field-aware interaction.
            shape: (batch_size, num_fields, emb_dim)
        """
        token_ffm_input, float_ffm_input = input_x[0], input_x[1] # [batch_size, num_fields]

        # get token field-aware embeddings
        token_input_x_emb = []
        if len(self.token_field_dims) > 0:
            token_input_x = token_ffm_input + token_ffm_input.new_tensor(self.token_offsets).unsqueeze(0)
            token_input_x_emb = [self.token_embeddings[i](token_input_x) for i in range(len(self.field_dims))] # [num_fields; [batch_size, num_token_field, emb_dim]]
        
        # get float field-aware embeddings
        float_input_x_emb = []
        if len(self.float_field_dims) > 0:
            index = torch.arange(0, self.float_num_fields).unsqueeze(0).expand_as(float_ffm_input).long().to(self.device) # [batch_size, num_float_fields]
            float_input_x_emb = [torch.mul(self.float_embeddings[i](index), float_ffm_input.unsqueeze(2)) for i in range(len(self.field_dims))] # [num_fields: [batch_size, num_float_fields, emb_dim]]
        
        input_x_emb = []  # [num_fields: [batch_size, num_fields, emb_dim]]
        if len(self.token_field_dims) > 0 and len(self.float_field_dims) > 0:
            for i in range(len(self.field_dims)):
                input_x_emb.append(torch.cat([token_input_x_emb[i], float_input_x_emb[i]], dim=1))
        elif len(self.token_field_dims) > 0:
            input_x_emb = token_input_x_emb
        elif len(self.float_field_dims) > 0:
            input_x_emb = float_input_x_emb
    
        output = list() # [batch_size, num_fields, emb_dim]
        for i in range(len(self.field_dims) - 1):
            for j in range(i + 1, len(self.field_dims)):
                output.append(input_x_emb[j][:, i] * input_x_emb[i][:, j])
        output = torch.stack(output, dim=1)

        return output
