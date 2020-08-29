# @Time   : 2020/8/6 14:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


"""
Reference:
Fuzheng Zhang et al., "Collaborative Knowledge Base Embedding for Recommender Systems." in SIGKDD 2016.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from ...utils import InputType
from ..abstract_recommender import KnowledgeRecommender
from ..loss import BPRLoss


# todo: L2 regularization
class CKE(KnowledgeRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(CKE, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)

        self.rec_loss = BPRLoss()
        self.kg_loss = BPRLoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        return self.item_embedding(item) + self.entity_embedding(item)

    def get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze()
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze()
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze()

        return h_e, r_e, pos_t_e, neg_t_e

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        user_e, pos_item_e = self.forward(user, pos_item)
        neg_item_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_item_e).sum(dim=1), torch.mul(user_e, neg_item_e).sum(dim=1)
        rec_loss = - self.rec_loss(pos_item_score, neg_item_score)

        h_e, r_e, pos_t_e, neg_t_e = self.get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score, neg_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1), ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = - self.kg_loss(pos_tail_score, neg_tail_score)

        return rec_loss + kg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)
