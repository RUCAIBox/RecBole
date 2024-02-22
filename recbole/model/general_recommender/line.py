# -*- coding: utf-8 -*-
# @Time   : 2020/12/8
# @Author : Yihong Guo
# @Email  : gyihong@hotmail.com

r"""
LINE
################################################
Reference:
    Jian Tang et al. "LINE: Large-scale Information Network Embedding." in WWW 2015.

Reference code:
    https://github.com/shenweichen/GraphEmbedding
"""

import random

import numpy as np
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class NegSamplingLoss(nn.Module):
    def __init__(self):
        super(NegSamplingLoss, self).__init__()

    def forward(self, sign, score):
        return -torch.mean(torch.log(torch.sigmoid(sign * score)))


class LINE(GeneralRecommender):
    r"""LINE is a graph embedding model.

    We implement the model to train users and items embedding for recommendation.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LINE, self).__init__(config, dataset)

        self.embedding_size = config["embedding_size"]
        self.order = config["order"]
        self.second_order_loss_weight = config["second_order_loss_weight"]

        self.interaction_feat = dataset.inter_feat

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        if self.order == 2:
            self.user_context_embedding = nn.Embedding(
                self.n_users, self.embedding_size
            )
            self.item_context_embedding = nn.Embedding(
                self.n_items, self.embedding_size
            )

        self.loss_fct = NegSamplingLoss()

        self.used_ids = self.get_used_ids()
        self.random_list = self.get_user_id_list()
        np.random.shuffle(self.random_list)
        self.random_pr = 0
        self.random_list_length = len(self.random_list)

        self.apply(xavier_normal_initialization)

    def get_used_ids(self):
        cur = np.array([set() for _ in range(self.n_items)])
        for uid, iid in zip(
            self.interaction_feat[self.USER_ID].numpy(),
            self.interaction_feat[self.ITEM_ID].numpy(),
        ):
            cur[iid].add(uid)
        return cur

    def sampler(self, key_ids):
        key_ids = np.array(key_ids.cpu())
        key_num = len(key_ids)
        total_num = key_num
        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        key_ids = np.tile(key_ids, 1)
        while len(check_list) > 0:
            value_ids[check_list] = self.random_num(len(check_list))
            check_list = np.array(
                [
                    i
                    for i, used, v in zip(
                        check_list,
                        self.used_ids[key_ids[check_list]],
                        value_ids[check_list],
                    )
                    if v in used
                ]
            )

        return torch.tensor(value_ids, device=self.device)

    def random_num(self, num):
        value_id = []
        self.random_pr %= self.random_list_length
        while True:
            if self.random_pr + num <= self.random_list_length:
                value_id.append(self.random_list[self.random_pr : self.random_pr + num])
                self.random_pr += num
                break
            else:
                value_id.append(self.random_list[self.random_pr :])
                num -= self.random_list_length - self.random_pr
                self.random_pr = 0
                np.random.shuffle(self.random_list)
        return np.concatenate(value_id)

    def get_user_id_list(self):
        return np.arange(1, self.n_users)

    def forward(self, h, t):
        h_embedding = self.user_embedding(h)
        t_embedding = self.item_embedding(t)

        return torch.sum(h_embedding.mul(t_embedding), dim=1)

    def context_forward(self, h, t, field):
        if field == "uu":
            h_embedding = self.user_embedding(h)
            t_embedding = self.item_context_embedding(t)
        else:
            h_embedding = self.item_embedding(h)
            t_embedding = self.user_context_embedding(t)

        return torch.sum(h_embedding.mul(t_embedding), dim=1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        score_pos = self.forward(user, pos_item)

        ones = torch.ones(len(score_pos), device=self.device)

        if self.order == 1:
            if random.random() < 0.5:
                score_neg = self.forward(user, neg_item)
            else:
                neg_user = self.sampler(pos_item)
                score_neg = self.forward(neg_user, pos_item)
            return self.loss_fct(ones, score_pos) + self.loss_fct(-1 * ones, score_neg)

        else:
            # randomly train i-i relation and u-u relation with u-i relation
            if random.random() < 0.5:
                score_neg = self.forward(user, neg_item)
                score_pos_con = self.context_forward(user, pos_item, "uu")
                score_neg_con = self.context_forward(user, neg_item, "uu")
            else:
                # sample negative user for item
                neg_user = self.sampler(pos_item)
                score_neg = self.forward(neg_user, pos_item)
                score_pos_con = self.context_forward(pos_item, user, "ii")
                score_neg_con = self.context_forward(pos_item, neg_user, "ii")

            return (
                self.loss_fct(ones, score_pos)
                + self.loss_fct(-1 * ones, score_neg)
                + self.loss_fct(ones, score_pos_con) * self.second_order_loss_weight
                + self.loss_fct(-1 * ones, score_neg_con)
                * self.second_order_loss_weight
            )

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        scores = self.forward(user, item)

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        # get user embedding from storage variable
        u_embeddings = self.user_embedding(user)
        i_embedding = self.item_embedding.weight
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, i_embedding.transpose(0, 1))

        return scores.view(-1)
