# -*- coding: utf-8 -*-
# @Time   : 2020/9/15
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
Reference:

"""

import torch
import torch.nn as nn

from ...utils import InputType
from ..abstract_recommender import KnowledgeRecommender
from ..loss import BPRLoss, EmbLoss
from ..utils import xavier_normal_initialization


class KGAT(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGAT, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.layers = [self.embedding_size] + config['layers']
        self.reg_weight = config['reg_weight']
        self.kg_weight = config['kg_weight']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        self.gnnlayers = nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            pass
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        posi_embeddings = entity_all_embeddings[pos_item]
        negi_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        reg_loss = self.reg_loss(u_embeddings, posi_embeddings, negi_embeddings)

        return mf_loss + self.reg_weight * reg_loss

    def predict(self):
        pass
