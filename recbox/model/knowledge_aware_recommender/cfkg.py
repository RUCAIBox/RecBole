# -*- coding: utf-8 -*-
# @Time   : 2020/9/14
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
Reference:
Qingyao Ai et al. "Learning heterogeneous knowledge base embeddings for explainable recommendation." in MDPI 2018.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from recbox.utils import InputType
from recbox.model.abstract_recommender import KnowledgeRecommender
from recbox.model.init import xavier_normal_initialization


"""
Todo:
The original paper puts rec and kg together for sampling and mix them for training.
Due to the limitation of the dataloader, it can only be sampled separately at present. 
Further improved.
"""


class CFKG(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CFKG, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.loss_function = config['loss_function']
        self.p_norm = config['p_norm']
        self.margin = config['margin']
        assert self.loss_function in ['inner_product', 'transe']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations + 1, self.embedding_size)
        self.rec_loss = nn.TripletMarginLoss(margin=self.margin, p=self.p_norm, reduction='mean') \
            if self.loss_function == 'transe' else InnerProductLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.entity_embedding(item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)
        return user_e, item_e, rec_r_e

    def get_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def get_kg_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e

    def get_score(self, h_e, r_e, t_e):
        if self.loss_function == 'transe':
            return - torch.norm(h_e + r_e - t_e, p=self.p_norm, dim=1)
        else:
            return torch.mul(h_e + r_e, t_e).sum(dim=1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        user_e, pos_item_e, neg_item_e, rec_r_e = self.get_rec_embedding(user, pos_item, neg_item)
        head_e, pos_tail_e, neg_tail_e, relation_e = self.get_kg_embedding(head, pos_tail, neg_tail, relation)

        h_e = torch.cat([user_e, head_e])
        r_e = torch.cat([rec_r_e, relation_e])
        pos_t_e = torch.cat([pos_item_e, pos_tail_e])
        neg_t_e = torch.cat([neg_item_e, neg_tail_e])

        loss = self.rec_loss(h_e + r_e, pos_t_e, neg_t_e)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e, rec_r_e = self.forward(user, item)
        score = self.get_score(user_e, item_e, rec_r_e)
        return score

    # def full_sort_predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     user_e = self.user_embedding(user)
    #     rec_r_e = self.relation_embedding.weight[-1]
    #     rec_r_e = rec_r_e.expand_as(user_e)
    #     all_item_e = self.entity_embedding.weight[:self.n_items]
    #     score = torch.matmul(user_e + rec_r_e, all_item_e.transpose(0, 1))
    #     return score.view(-1)


class InnerProductLoss(nn.Module):

    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        pos_score = torch.mul(anchor, positive).sum(dim=1)
        neg_score = torch.mul(anchor, negative).sum(dim=1)
        return (F.softplus(- pos_score) + F.softplus(neg_score)).mean()
