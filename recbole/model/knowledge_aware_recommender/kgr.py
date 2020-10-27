# -*- coding: utf-8 -*-
# @Time   : 2020/10/25
# @Author : Jin Huang and Shanlei Mu
# @Email  : Betsyj.huang@gmail.com and slmu@ruc.edu.cn

r"""
KGR
##################################################
Reference:
    Knowledge-enhanced General Recommendation. TBD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.utils import InputType
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_normal_initialization


class KGR(KnowledgeRecommender):
    r"""KGR is a knowledge-enhanced general recommendation model, it can incorporate KG and other information such as corresponding
    images to enrich the representation of items for item recommendations.

    Note:
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGR, self).__init__(config, dataset)

        # load dataset info
        # self.entity_embedding_matrix = dataset.get_preload_weight('ent_id')
        # self.relation_embedding_matrix = dataset.get_preload_weight('rel_id')
        # Todo: Without relative docs, I randomly generate some data. (See below)
        import numpy as np
        self.embedding_size = config['embedding_size']
        self.entity_embedding_matrix = np.random.randn(self.n_entities, self.embedding_size)
        self.relation_embedding_matrix = np.random.randn(self.n_relations, self.embedding_size)


        # load parameters info
        self.embedding_size = config['embedding_size']
        self.freeze_kg = config['freeze_kg']
        self.reg_weight = config['reg_weight']
        self.gamma = 10

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_memory_embedding = nn.Embedding(self.n_users, self.n_relations * self.embedding_size) # U*(R*E)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_embedding.weight.requires_grad = not self.freeze_kg

        self.dense_layer_u = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.dense_layer_i = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.entity_embedding.weight.data.copy_(torch.from_numpy(self.entity_embedding_matrix))
        self.relation_Matrix = torch.from_numpy(self.relation_embedding_matrix[:self.n_relations]) # R*E
        
    def _get_kg_embedding(self, head):
        # Difference: We generate the embeddings of the tail entities on every relations only for head due to the 1-N problems.
        head_e = self.entity_embedding(head) # B*E
        # # use all relation embs as the relation matrix, M_R, self.relation_embedding
        # rels = torch.LongTensor([x for x in range(self.n_relations)])
        # relation_Matrix = self.relation_embedding[rels] # R*E
        # print(head.size(), head_e.unsqueeze(1).size())
        relation_Matrix = self.relation_Matrix.repeat(head_e.size()[0], 1, 1) # B*R*E
        head_Matrix = head_e.unsqueeze(1).repeat(1, self.n_relations, 1) # B*R*E
        tail_Matrix = head_Matrix + relation_Matrix
        
        return head_e, tail_Matrix

    def forward(self, user, item):
        u_e = self.user_embedding(user)
        i_e = self.item_embedding(item)
        u_memory = self.user_memory_embedding(user).view(-1, self.n_relations, self.embedding_size) # B*R*E
        h_e, i_memory = self._get_kg_embedding(item) # B*R*E
        # Memory Read Operation (attentive combination)
        attentions = nn.functional.softmax(self.gamma * torch.mul(u_memory, i_memory).sum(-1).float(), -1) # B*R
        u_m = torch.mul(u_memory, attentions.unsqueeze(-1)).sum(1) # B*R*E times B*R*1 -> B*E

        p_u = self.dense_layer_u(torch.cat((u_e, u_m), -1))  # B*(2E) -> B*E
        q_i = self.dense_layer_i(torch.cat((i_e, h_e), -1))  # B*(2E) -> B*E

        # Here it could be different from KSR, where u_m (user current preference on KG depends on the last item. But u_m depends on item, no matter pos or neg item.)
        score = torch.mul(p_u, q_i).sum(dim=1)
        return score

    # def _get_rec_loss(self, user_e, pos_e, neg_e):
    #     pos_score = torch.mul(user_e, pos_e).sum(dim=1)
    #     neg_score = torch.mul(user_e, neg_e).sum(dim=1)
    #     rec_loss = self.rec_loss(pos_score, neg_score)
    #     return rec_loss

    # def _get_kg_loss(self, h_e, r_e, pos_e, neg_e):
    #     pos_tail_score = ((h_e + r_e - pos_e) ** 2).sum(dim=1)
    #     neg_tail_score = ((h_e + r_e - neg_e) ** 2).sum(dim=1)
    #     kg_loss = self.kg_loss(neg_tail_score, pos_tail_score)
    #     return kg_loss

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        pos_score = self.forward(user, pos_item)
        neg_score = self.forward(user, neg_item)
        rec_loss = self.rec_loss(pos_score, neg_score)

        user_e = self.user_embedding(user)
        pos_item_e = self.item_embedding(pos_item)
        neg_item_e = self.item_embedding(neg_item)
        reg_loss = self.reg_weight * self.reg_loss(user_e, pos_item_e, neg_item_e)
        return rec_loss, reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID] # B
        test_item = torch.LongTensor([x for x in range(self.n_items)]) # N
        score = self.forward(user.unsqueeze(1).repeat(1, self.n_items).view(-1), test_item.unsqueeze(0).repeat(user.size()[0], 1).view(-1)) # (B*N), (B*N)
        return score.view(user.size()[0], -1)
