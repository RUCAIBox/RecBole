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

from ..layers import Dense, CrossCompressUnit

from recbox.utils import InputType
from recbox.model.abstract_recommender import KnowledgeRecommender
from recbox.model.loss import BPRLoss
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

        self.n_user = self.n_users
        self.n_item = self.n_items
        self.n_entity = self.n_entities
        self.n_relation = self.n_relations
        self.n_heads = dataset.num(self.HEAD_ENTITY_ID)
        self.n_tails = dataset.num(self.TAIL_ENTITY_ID)

        self._build_model()
        self._build_loss()

    def _build_model(self):
        print("Build models")
        self.MKR_model = MKR_model(self.n_user, self.n_item, self.n_entity, self.n_relation, self.embedding_size, self.L, self.H)
        self.MKR_model = self.MKR_model.to(self.device, non_blocking=True)
        for m in self.MKR_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)

    def _build_loss(self):
        self.sigmoid_BCE = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.rec_loss = BPRLoss()

    def _inference_rs(self, interaction):
        # Inputs
        self.user_indices = torch.cat([interaction[self.USER_ID], interaction[self.USER_ID]], 0)
        self.item_indices = torch.cat([interaction[self.ITEM_ID], interaction[self.NEG_ITEM_ID]], 0)
        self.head_indices = torch.cat([interaction[self.ITEM_ID], interaction[self.NEG_ITEM_ID]], 0)
        self.labels = torch.cat([torch.ones_like(interaction[self.ITEM_ID], dtype=float), \
                                torch.zeros_like(interaction[self.NEG_ITEM_ID], dtype=float)], 0)
        
        # Inference
        outputs = self.MKR_model(user_indices=self.user_indices,
                                 item_indices=self.item_indices,
                                 head_indices=self.head_indices,
                                 relation_indices=None,
                                 tail_indices=None)

        user_embeddings, item_embeddings, scores, scores_normalized = outputs

        return user_embeddings, item_embeddings, scores, scores_normalized, self.labels

    def _inference_kge(self, interaction):
        # Inputs
        self.item_indices = interaction[self.HEAD_ENTITY_ID]
        self.head_indices = interaction[self.HEAD_ENTITY_ID]
        self.relation_indices = interaction[self.RELATION_ID]
        self.tail_indices = interaction[self.TAIL_ENTITY_ID]

        # Inference
        outputs = self.MKR_model(user_indices=None,
                                 item_indices=self.item_indices,
                                 head_indices=self.head_indices,
                                 relation_indices=self.relation_indices,
                                 tail_indices=self.tail_indices)

        head_embeddings, tail_embeddings, scores_kge, rmse = outputs
        return head_embeddings, tail_embeddings, scores_kge, rmse

    def l2_loss(self, inputs):
        return torch.sum(inputs ** 2) / 2

    def loss_rs(self, user_embeddings, item_embeddings, scores, labels):

        base_loss_rs = torch.mean(self.sigmoid_BCE(scores, labels))
        l2_loss_rs = self.l2_loss(user_embeddings) + self.l2_loss(item_embeddings)
        '''
        for name, param in self.MKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('rs' in name) or ('cc_unit' in name) or ('user' in name)) \
                    and ('weight' in name):
                l2_loss_rs = l2_loss_rs + self.l2_loss(param)
        '''
        loss_rs = base_loss_rs + l2_loss_rs * self.l2_weight

        return loss_rs, base_loss_rs, l2_loss_rs

    def loss_kge(self, scores_kge, head_embeddings, tail_embeddings):
        base_loss_kge = -scores_kge
        l2_loss_kge = self.l2_loss(head_embeddings) + self.l2_loss(tail_embeddings)
        '''
        for name, param in self.MKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('kge' in name) or ('tail' in name) or ('cc_unit' in name)) \
                    and ('weight' in name):
                l2_loss_kge = l2_loss_kge + self.l2_loss(param)
        '''
        # Note: L2 regularization will be done by weight_decay of pytorch optimizer
        loss_kge = base_loss_kge + l2_loss_kge * self.l2_weight

        return loss_kge, base_loss_kge, l2_loss_kge

    
    def calculate_rs_loss(self, interaction):
        self.MKR_model.train()
        user_embeddings, item_embeddings, scores, _, labels= self._inference_rs(interaction)
        loss_rs, base_loss_rs, l2_loss_rs = self.loss_rs(user_embeddings, item_embeddings, scores, labels)

        return user_embeddings, item_embeddings, item_embeddings, scores, labels, loss_rs

    def calculate_kg_loss(self, interaction):
        self.MKR_model.train()
        head_embeddings, tail_embeddings, scores_kge, rmse = self._inference_kge(interaction)
        loss_kge, base_loss_kge, l2_loss_kge = self.loss_kge(scores_kge, head_embeddings, tail_embeddings)

        return head_embeddings, tail_embeddings, scores_kge, rmse, loss_kge

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        head = interaction[self.ITEM_ID]

        self.MKR_model.train()
        outputs = self.MKR_model(user, item, head, None, None)
        _, _, scores, _ = outputs

        return scores

class MKR_model(nn.Module):
    r"""This module contains the initialization and implement of MKR model.
    """
    def __init__(self, n_user, n_item, n_entity, n_relation, embedding_size, L, H, use_inner_product=True):
        super(MKR_model, self).__init__()

        # <Lower Model>
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.embedding_size = embedding_size
        self.L = L
        self.H = H
        self.use_inner_product = use_inner_product

        # Init embeddings
        self.user_embeddings_lookup = nn.Embedding(self.n_user, self.embedding_size)
        self.item_embeddings_lookup = nn.Embedding(self.n_entity, self.embedding_size)
        self.entity_embeddings_lookup = nn.Embedding(self.n_entity, self.embedding_size)
        self.relation_embeddings_lookup = nn.Embedding(self.n_relation, self.embedding_size)

        self.user_mlp = nn.Sequential()
        self.tail_mlp = nn.Sequential()
        self.cc_unit = nn.Sequential()
        for i_cnt in range(self.L):
            self.user_mlp.add_module('user_mlp{}'.format(i_cnt),
                                     Dense(self.embedding_size, self.embedding_size))
            self.tail_mlp.add_module('tail_mlp{}'.format(i_cnt),
                                     Dense(self.embedding_size, self.embedding_size))
            self.cc_unit.add_module('cc_unit{}'.format(i_cnt),
                                     CrossCompressUnit(self.embedding_size))
        # <Higher Model>
        self.kge_pred_mlp = Dense(self.embedding_size * 2, self.embedding_size)
        self.kge_mlp = nn.Sequential()
        for i_cnt in range(self.H - 1):
            self.kge_mlp.add_module('kge_mlp{}'.format(i_cnt),
                                    Dense(self.embedding_size * 2, self.embedding_size * 2))
        if self.use_inner_product==False:
            self.rs_pred_mlp = Dense(self.embedding_size * 2, 1)
            self.rs_mlp = nn.Sequential()
            for i_cnt in range(self.H - 1):
                self.rs_mlp.add_module('rs_mlp{}'.format(i_cnt),
                                       Dense(self.embedding_size * 2, self.embedding_size * 2))

    def forward(self, user_indices=None, item_indices=None, head_indices=None,
            relation_indices=None, tail_indices=None):

        # <Lower Model>
        if user_indices is not None:
            self.user_indices = user_indices
        if item_indices is not None:
            self.item_indices = item_indices
        if head_indices is not None:
            self.head_indices = head_indices
        if relation_indices is not None:
            self.relation_indices = relation_indices
        if tail_indices is not None:
            self.tail_indices = tail_indices

        # Embeddings
        self.item_embeddings = self.item_embeddings_lookup(self.item_indices)
        self.head_embeddings = self.entity_embeddings_lookup(self.head_indices)
        self.item_embeddings, self.head_embeddings = self.cc_unit([self.item_embeddings, self.head_embeddings])

        # <Higher Model>
        if user_indices is not None:
            # RS
            self.user_embeddings = self.user_embeddings_lookup(self.user_indices)
            self.user_embeddings = self.user_mlp(self.user_embeddings)
            
            if self.use_inner_product:
                # [batch_size]
                self.scores = torch.sum(self.user_embeddings * self.item_embeddings, 1)
            else:
                # [batch_size, dim * 2]
                self.user_item_concat = torch.cat([self.user_embeddings, self.item_embeddings], 1)
                self.user_item_concat = self.rs_mlp(self.user_item_concat)
                # [batch_size]
                self.scores = torch.squeeze(self.rs_pred_mlp(self.user_item_concat))
            self.scores_normalized = torch.sigmoid(self.scores)
            outputs = [self.user_embeddings, self.item_embeddings, self.scores, self.scores_normalized]

        if relation_indices is not None:
            # KGE
            self.tail_embeddings = self.entity_embeddings_lookup(self.tail_indices)
            self.relation_embeddings = self.relation_embeddings_lookup(self.relation_indices)
            self.tail_embeddings = self.tail_mlp(self.tail_embeddings)
            # [batch_size, dim * 2]
            self.head_relation_concat = torch.cat([self.head_embeddings, self.relation_embeddings], 1)
            self.head_relation_concat = self.kge_mlp(self.head_relation_concat)
            # [batch_size, 1]
            self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
            self.tail_pred = torch.sigmoid(self.tail_pred)
            self.scores_kge = torch.sigmoid(torch.sum(self.tail_embeddings * self.tail_pred, 1))
            self.rmse = torch.mean(
                torch.sqrt(torch.sum(torch.pow(self.tail_embeddings -
                           self.tail_pred, 2), 1) / self.embedding_size))
            outputs = [self.head_embeddings, self.tail_embeddings, self.scores_kge, self.rmse]

        return outputs


