# -*- coding: utf-8 -*-
# @Time   : 2020/10/08
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn

r"""
recbox.model.knowledge_aware_recommender.mkr
#####################################################
Reference:
Hongwei Wang et al. "Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation." in WWW 2019.

Reference code:
https://github.com/hsientzucheng/MKR.PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbox.utils import InputType
from recbox.model.abstract_recommender import KnowledgeRecommender
from recbox.model.init import xavier_normal_initialization

class MKR(KnowledgeRecommender):
    r"""MKR is a Multi-task feature learning approach for Knowledge graph enhanced Recommendation.
    It is a deep end-to-end framework that utilizes knowledge graph embedding task to assist recommendation task. 
    The two tasks are associated by cross&compress units, which automatically share latent features and 
    learn high-order interactions between items in recommender systems and entities in the knowledge graph.
    """

    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(MKR, self).__init__(config, dataset)
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.L = config['low_layers_num'] # the number of low layers
        self.H = config['high_layers_num'] # the number of high layers
        self.l2_weight = config['l2_weight']
        self.use_inner_product = config['use_inner_product']

        # init embeddings
        self.user_embeddings_lookup = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embeddings_lookup = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_embeddings_lookup = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embeddings_lookup = nn.Embedding(self.n_relations, self.embedding_size)

        # define layers
        self.user_mlp = nn.Sequential()
        self.tail_mlp = nn.Sequential()
        self.cc_unit = nn.Sequential()
        self.kge_mlp = nn.Sequential()
        self.kge_pred_mlp = Dense(self.embedding_size * 2, self.embedding_size)
        self._init_layers()
        
        # loss
        self.sigmoid_BCE = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        
    def _init_layers(self):
        r"""Initial lower layers and higher layers of MKR model.

        """
        for i_cnt in range(self.L):
            self.user_mlp.add_module('user_mlp{}'.format(i_cnt), Dense(self.embedding_size, self.embedding_size))
            self.tail_mlp.add_module('tail_mlp{}'.format(i_cnt), Dense(self.embedding_size, self.embedding_size))
            self.cc_unit.add_module('cc_unit{}'.format(i_cnt), CrossCompressUnit(self.embedding_size))
        for i_cnt in range(self.H - 1):
            self.kge_mlp.add_module('kge_mlp{}'.format(i_cnt), Dense(self.embedding_size * 2, self.embedding_size * 2))
        if self.use_inner_product == False:
            self.rs_pred_mlp = Dense(self.embedding_size * 2, 1)
            self.rs_mlp = nn.Sequential()
            for i_cnt in range(self.H - 1):
                self.rs_mlp.add_module('rs_mlp{}'.format(i_cnt), Dense(self.embedding_size * 2, self.embedding_size * 2))

    def forward(self, user_indices=None, item_indices=None, head_indices=None,
                relation_indices=None, tail_indices=None):
        self.item_embeddings = self.item_embeddings_lookup(item_indices)
        self.head_embeddings = self.entity_embeddings_lookup(head_indices)
        self.item_embeddings, self.head_embeddings = self.cc_unit([self.item_embeddings, self.head_embeddings]) # calculate feature interactions between items and entities

        if user_indices is not None:
            # RS
            self.user_embeddings = self.user_embeddings_lookup(user_indices)
            self.user_embeddings = self.user_mlp(self.user_embeddings)
            
            if self.use_inner_product: # get scores by inner product.
                self.scores = torch.sum(self.user_embeddings * self.item_embeddings, 1) # [batch_size]
            else: # get scores by mlp layers
                self.user_item_concat = torch.cat([self.user_embeddings, self.item_embeddings], 1) # [batch_size, emb_dim*2]
                self.user_item_concat = self.rs_mlp(self.user_item_concat)
                
                self.scores = torch.squeeze(self.rs_pred_mlp(self.user_item_concat)) # [batch_size]
            self.scores_normalized = torch.sigmoid(self.scores)
            outputs = [self.user_embeddings, self.item_embeddings, self.scores, self.scores_normalized]

        if relation_indices is not None:
            # KGE
            self.tail_embeddings = self.entity_embeddings_lookup(tail_indices)
            self.relation_embeddings = self.relation_embeddings_lookup(relation_indices)
            self.tail_embeddings = self.tail_mlp(self.tail_embeddings)
            
            self.head_relation_concat = torch.cat([self.head_embeddings, self.relation_embeddings], 1) # [batch_size, emb_dim*2]
            self.head_relation_concat = self.kge_mlp(self.head_relation_concat)

            self.tail_pred = self.kge_pred_mlp(self.head_relation_concat) # [batch_size, 1]
            self.tail_pred = torch.sigmoid(self.tail_pred)
            self.scores_kge = torch.sigmoid(torch.sum(self.tail_embeddings * self.tail_pred, 1))
            self.rmse = torch.mean(
                torch.sqrt(torch.sum(torch.pow(self.tail_embeddings -
                           self.tail_pred, 2), 1) / self.embedding_size))
            outputs = [self.head_embeddings, self.tail_embeddings, self.scores_kge, self.rmse]

        return outputs

    def l2_loss(self, inputs):
        return torch.sum(inputs ** 2) / 2

    def calculate_rs_loss(self, interaction):
        r"""Calculate the training loss for a batch data of RS.

        """
        # inputs
        self.user_indices = torch.cat([interaction[self.USER_ID], interaction[self.USER_ID]], 0)
        self.item_indices = torch.cat([interaction[self.ITEM_ID], interaction[self.NEG_ITEM_ID]], 0)
        self.head_indices = torch.cat([interaction[self.ITEM_ID], interaction[self.NEG_ITEM_ID]], 0)
        self.labels = torch.cat([torch.ones_like(interaction[self.ITEM_ID], dtype=float), \
                                torch.zeros_like(interaction[self.NEG_ITEM_ID], dtype=float)], 0)
        # RS model
        user_embeddings, item_embeddings, \
        scores, scores_normalized = self.forward(user_indices=self.user_indices,
                                                 item_indices=self.item_indices,
                                                 head_indices=self.head_indices,
                                                 relation_indices=None,
                                                 tail_indices=None)
        # loss
        base_loss_rs = torch.mean(self.sigmoid_BCE(scores, self.labels))
        l2_loss_rs = self.l2_loss(user_embeddings) + self.l2_loss(item_embeddings)
        loss_rs = base_loss_rs + l2_loss_rs * self.l2_weight

        return user_embeddings, item_embeddings, scores, self.labels, loss_rs

    def calculate_kg_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.

        """
        # inputs
        self.item_indices = interaction[self.HEAD_ENTITY_ID]
        self.head_indices = interaction[self.HEAD_ENTITY_ID]
        self.relation_indices = interaction[self.RELATION_ID]
        self.tail_indices = interaction[self.TAIL_ENTITY_ID]
        # KGE model
        head_embeddings, tail_embeddings, \
        scores_kge, rmse = self.forward(user_indices=None,
                                        item_indices=self.item_indices,
                                        head_indices=self.head_indices,
                                        relation_indices=self.relation_indices,
                                        tail_indices=self.tail_indices)
        # loss
        base_loss_kge = -scores_kge
        l2_loss_kge = self.l2_loss(head_embeddings) + self.l2_loss(tail_embeddings)
        loss_kge = base_loss_kge + l2_loss_kge * self.l2_weight

        return head_embeddings, tail_embeddings, scores_kge, rmse, loss_kge

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        head = interaction[self.ITEM_ID]

        outputs = self.forward(user, item, head)
        _, _, scores, _ = outputs

        return scores

class Dense(nn.Module):
    r"""This is MLP layers for MKR model.

    """
    def __init__(self, input_dim, output_dim, dropout=0.0, chnl=8):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        
        return self.act(output)

class CrossCompressUnit(nn.Module):
    r"""This is Cross&Compress Unit for MKR model to model feature interactions between items and entities.

    """
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = nn.Linear(dim, 1, bias=True)
        self.fc_ev = nn.Linear(dim, 1, bias=True)
        self.fc_ve = nn.Linear(dim, 1, bias=True)
        self.fc_ee = nn.Linear(dim, 1, bias=True)

    def forward(self, inputs):
        v, e = inputs
        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)
        # [batch_size, dim, dim]
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0,2,1)
        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)
        # [batch_size, dim]
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_output = v_intermediate.view(-1, self.dim)
        e_output = e_intermediate.view(-1, self.dim)

        return v_output, e_output

