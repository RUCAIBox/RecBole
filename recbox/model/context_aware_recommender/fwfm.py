# -*- coding: utf-8 -*-
# @Time   : 2020/10/06
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn
# @File   : fwfm.py

r"""
recbox.model.context_aware_recommender.fwfm
#####################################################
Reference:
Junwei Pan et al. "Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising." in WWW 2018.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbox.model.layers import FieldAwareFactorizationMachine
from recbox.model.context_aware_recommender.context_recommender import ContextRecommender


class FwFM(ContextRecommender):
    r"""FwFM is a context-based recommendation model.
    It aims to model the different feature interactions between different fields in a much more memory-efficient way.
    It proposes a field pair weight matrix r_{F(i),F(j)}, to capture the heterogeneity of field pair interactions.
    The model defines as follows:
    .. math::
       y = w_0 + \sum_{i=1}^{m}x_{i}w_{i} + \sum_{i=1}^{m}\sum_{j=i+1}^{m}x_{i}x_{j}<v_{i}, v_{j}>r_{F(i),F(j)}
    """
    def __init__(self, config, dataset):
        super(FwFM, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.dropout = config['dropout']
        self.num_pair = int(self.num_feature_field * (self.num_feature_field-1) / 2)
        self.dropout_layer = nn.Dropout(p=self.dropout)
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
    
    def build_cross(self, feat_emb):
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]  # [batch_size, num_pairs, emb_dim]
        q = feat_emb[:, col]  # [batch_size, num_pairs, emb_dim]
        
        return p, q

    def fwfm_layer(self, infeature):
        r"""Get the field pair weight matrix r_{F(i),F(j)}.
        And model the different interaction strengths of different field pairs 
        \sum_{i=1}^{m}\sum_{j=i+1}^{m}x_{i}x_{j}<v_{i}, v_{j}>r_{F(i),F(j)}.
        Args:
            infeature (torch.cuda.FloatTensor): (batch_size,field_size,embed_dim)

        Returns:
            torch.cuda.FloatTensor: (batch_size,1)
        """

        p, q = self.build_cross(infeature)
        pair_wise_inter = torch.mul(p, q)  # [batch_size, num_pairs, emb_dim]

        # get r(Fi, Fj), [batch_size, num_pair, emb_dim]
        batch_size = infeature.shape[0]
        para = torch.randn(self.num_pair*self.embedding_size).expand(batch_size, self.num_pair*self.embedding_size).to(self.device) # [batch_size, num_pairs*emb_dim]
        para = torch.reshape(para, (batch_size, self.num_pair, self.embedding_size))
        r = nn.Parameter(para, requires_grad=True)
        
        fwfm_inter = torch.mul(r, pair_wise_inter) # [batch_size, num_pairs, emb_dim]
        fwfm_inter = torch.sum(fwfm_inter, dim=1) # [batch_size, emb_dim]
        fwfm_inter = self.dropout_layer(fwfm_inter)  

        fwfm_output = torch.sum(fwfm_inter, dim=1, keepdim=True)  # [batch_size, 1]
        return fwfm_output

    def forward(self, interaction):
        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        fwfm_all_embeddings = torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]

        output = self.sigmoid(self.first_order_linear(interaction) + self.fwfm_layer(fwfm_all_embeddings))

        return output.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
