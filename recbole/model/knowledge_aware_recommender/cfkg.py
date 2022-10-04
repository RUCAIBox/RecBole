# -*- coding: utf-8 -*-
# @Time   : 2020/9/14
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
CFKG
##################################################
Reference:
    Qingyao Ai et al. "Learning heterogeneous knowledge base embeddings for explainable recommendation." in MDPI 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class CFKG(KnowledgeRecommender):
    r"""CFKG is a knowledge-based recommendation model, it combines knowledge graph and the user-item interaction
    graph to a new graph. In this graph, user, item and related attribute are viewed as entities, and the interaction
    between user and item and the link between item and attribute are viewed as relations. It define a new score
    function as follows:

    .. math::
        d (u_i + r_{buy}, v_j)

    Note:
        In the original paper, CFKG puts recommender data (u-i interaction) and knowledge data (h-r-t) together
        for sampling and mix them for training. In this version, we sample recommender data
        and knowledge data separately, and put them together for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CFKG, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.loss_function = config["loss_function"]
        self.margin = config["margin"]
        assert self.loss_function in ["inner_product", "transe"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(
            self.n_relations + 1, self.embedding_size
        )
        if self.loss_function == "transe":
            self.rec_loss = nn.TripletMarginLoss(
                margin=self.margin, p=2, reduction="mean"
            )
        else:
            self.rec_loss = InnerProductLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.entity_embedding(item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)
        score = self._get_score(user_e, item_e, rec_r_e)
        return score

    def _get_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_kg_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e

    def _get_score(self, h_e, t_e, r_e):
        if self.loss_function == "transe":
            return -torch.norm(h_e + r_e - t_e, p=2, dim=1)
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

        user_e, pos_item_e, neg_item_e, rec_r_e = self._get_rec_embedding(
            user, pos_item, neg_item
        )
        head_e, pos_tail_e, neg_tail_e, relation_e = self._get_kg_embedding(
            head, pos_tail, neg_tail, relation
        )

        h_e = torch.cat([user_e, head_e])
        r_e = torch.cat([rec_r_e, relation_e])
        pos_t_e = torch.cat([pos_item_e, pos_tail_e])
        neg_t_e = torch.cat([neg_item_e, neg_tail_e])

        loss = self.rec_loss(h_e + r_e, pos_t_e, neg_t_e)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)


class InnerProductLoss(nn.Module):
    r"""This is the inner-product loss used in CFKG for optimization."""

    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        pos_score = torch.mul(anchor, positive).sum(dim=1)
        neg_score = torch.mul(anchor, negative).sum(dim=1)
        return (F.softplus(-pos_score) + F.softplus(neg_score)).mean()
