# -*- coding: utf-8 -*-
# @Time   : 2020/8/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
CKE
##################################################
Reference:
    Fuzheng Zhang et al. "Collaborative Knowledge Base Embedding for Recommender Systems." in SIGKDD 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class CKE(KnowledgeRecommender):
    r"""CKE is a knowledge-based recommendation model, it can incorporate KG and other information such as corresponding
    images to enrich the representation of items for item recommendations.

    Note:
        In the original paper, CKE used structural knowledge, textual knowledge and visual knowledge. In our
        implementation, we only used structural knowledge. Meanwhile, the version we implemented uses a simpler
        regular way which can get almost the same result (even better) as the original regular way.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CKE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.kg_embedding_size = config["kg_embedding_size"]
        self.reg_weights = config["reg_weights"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(
            self.n_relations, self.embedding_size * self.kg_embedding_size
        )
        self.rec_loss = BPRLoss()
        self.kg_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        r_e = F.normalize(r_e, p=2, dim=1)
        h_e = F.normalize(h_e, p=2, dim=1)
        pos_t_e = F.normalize(pos_t_e, p=2, dim=1)
        neg_t_e = F.normalize(neg_t_e, p=2, dim=1)

        return h_e, r_e, pos_t_e, neg_t_e, r_trans_w

    def forward(self, user, item):
        u_e = self.user_embedding(user)
        i_e = self.item_embedding(item) + self.entity_embedding(item)
        score = torch.mul(u_e, i_e).sum(dim=1)
        return score

    def _get_rec_loss(self, user_e, pos_e, neg_e):
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        rec_loss = self.rec_loss(pos_score, neg_score)
        return rec_loss

    def _get_kg_loss(self, h_e, r_e, pos_e, neg_e):
        pos_tail_score = ((h_e + r_e - pos_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_e) ** 2).sum(dim=1)
        kg_loss = self.kg_loss(neg_tail_score, pos_tail_score)
        return kg_loss

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        user_e = self.user_embedding(user)
        pos_item_e = self.item_embedding(pos_item)
        neg_item_e = self.item_embedding(neg_item)
        pos_item_kg_e = self.entity_embedding(pos_item)
        neg_item_kg_e = self.entity_embedding(neg_item)
        pos_item_final_e = pos_item_e + pos_item_kg_e
        neg_item_final_e = neg_item_e + neg_item_kg_e

        rec_loss = self._get_rec_loss(user_e, pos_item_final_e, neg_item_final_e)

        h_e, r_e, pos_t_e, neg_t_e, r_trans_w = self._get_kg_embedding(
            h, r, pos_t, neg_t
        )
        kg_loss = self._get_kg_loss(h_e, r_e, pos_t_e, neg_t_e)

        reg_loss = self.reg_weights[0] * self.reg_loss(
            user_e, pos_item_final_e, neg_item_final_e
        ) + self.reg_weights[1] * self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)

        return rec_loss, kg_loss, reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = (
            self.item_embedding.weight + self.entity_embedding.weight[: self.n_items]
        )
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
