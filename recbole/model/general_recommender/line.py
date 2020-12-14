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

import torch
import torch.nn as nn
import networkx as nx
import pandas as pd
import random

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization

class neg_sampling_loss(nn.Module):
    def __init__(self):
        super(neg_sampling_loss, self).__init__()

    def forward(self,score,sign):
        return -torch.mean(torch.sigmoid(sign * score))

class LINE(GeneralRecommender):
    r"""LINE is a graph embedding model.

    We implement the model to train users and items embedding for recommendation.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LINE, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.order = config['order']
        self.second_order_loss_weight = config['second_order_loss_weight']


        self.interaction_feat = dataset.dataset.inter_feat

        self.uid_field = dataset.dataset.uid_field
        self.iid_field = dataset.dataset.iid_field

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size).to(self.device)

        self.user_context_embedding = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.item_context_embedding = nn.Embedding(self.n_items, self.embedding_size).to(self.device)

        self.loss_fct = neg_sampling_loss()

        # graph initialization
        self.process_nodeid()
        self.read_graph()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def process_nodeid(self):

        u = self.interaction_feat[self.uid_field].numpy()
        i = self.interaction_feat[self.iid_field].numpy()

        interaction_feat_new = pd.DataFrame([])
        interaction_feat_new[self.uid_field] = u
        interaction_feat_new[self.iid_field] = i
        self.interaction_feat_new = interaction_feat_new

        self.interaction_feat_new[self.uid_field] = self.interaction_feat_new[self.uid_field].map(lambda x: "u" + str(x))
        self.interaction_feat_new[self.iid_field] = self.interaction_feat_new[self.iid_field].map(lambda x: "i" + str(x))

    def read_graph(self):
        self.g = nx.from_pandas_edgelist(self.interaction_feat_new,self.uid_field,self.iid_field)

    def generate_neg_user(self,item_id):

        neigh = list(self.g['i' + str(int(item_id))])

        curr = random.randint(1, self.n_users-1)
        while 'u' + str(curr) in neigh:
            curr = random.randint(1, self.n_users-1)

        return curr

    def gen_neg_sample(self,src_):

        t = []

        src = src_.cpu()

        for i in range(len(src)):
            t.append(self.generate_neg_user(src[i]))

        return src_,torch.LongTensor(t).to(self.device)

    def forward(self,h,t):

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

        score_pos = self.forward(user,pos_item)

        if random.random()<0.5:
            score_neg = self.forward(user, neg_item)
            score_pos_con = self.context_forward(user, pos_item, 'uu')
            score_neg_con = self.context_forward(user, neg_item, 'uu')
        else:
            h,t = self.gen_neg_sample(pos_item)
            score_neg = self.forward(t,h)
            score_pos_con = self.context_forward(h, user,'ii')
            score_neg_con = self.context_forward(h, t,'ii')

        ones = torch.ones(len(score_pos),device=self.device)
        if self.order == 1:
            return self.loss_fct(ones,score_pos) \
               + self.loss_fct(-1 * ones, score_neg)
        else:
            return self.loss_fct(ones,score_pos) \
                   + self.loss_fct(-1 * ones, score_neg)\
                   + self.loss_fct(ones,score_pos_con)*self.second_order_loss_weight\
                   + self.loss_fct(-1*ones,score_neg_con)*self.second_order_loss_weight

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
