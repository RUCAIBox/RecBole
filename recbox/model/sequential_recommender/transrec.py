# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 17:01
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
recbox.model.sequential_recommender.transrec
################################################

Reference:
Ruining He et al. "Translation-based Recommendation." In RecSys 2017.

"""

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, xavier_normal_
from recbox.utils import InputType
from recbox.model.abstract_recommender import SequentialRecommender
from recbox.model.loss import BPRLoss, EmbLoss, RegLoss
from recbox.model.init import xavier_normal_initialization


class TransRec(SequentialRecommender):
    r"""
    TransRec is translation-based model for sequential recommendation.
    It assumes that the `prev. item` + `user`  = `next item`.
    We use the Euclidean Distance to calculate the similarity in this implementation.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(TransRec, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.item_num = dataset.item_num
        self.user_num = dataset.user_num

        # embedding_size is equal to hidden_size
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        self.bias = nn.Embedding(self.item_num, 1, padding_idx=0) # Beta popularity bias
        self.T = nn.Parameter(torch.zeros(self.hidden_size)) # average user representation 'global'

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_loss = RegLoss()
        self.apply(xavier_normal_initialization)


    def l2_distance(self, x, y):
        return torch.sqrt(torch.sum((x - y)**2, dim=-1, keepdim=True)) # [B 1]

    def gather_last_items(self, item_list, gather_index):
        "Gathers the last_item at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1)
        last_items = item_list.gather(index=gather_index, dim=1) # [B 1]
        return last_items.squeeze(-1) # [B]

    def forward(self, interaction):
        user = interaction[self.USER_ID] # [B]
        items_list = interaction[self.ITEM_ID_LIST] # [B Len]
        # the last item at the last position
        last_items = self.gather_last_items(items_list, interaction[self.ITEM_LIST_LEN] - 1) # [B]

        user_emb = self.user_embedding(user) # [B H]
        last_items_emb = self.item_embedding(last_items)  # [B H]
        T = self.T.expand_as(user_emb) # [B H]
        output = user_emb + T + last_items_emb # [B H]
        return output


    def calculate_loss(self, interaction):
        output = self.forward(interaction) # [B H]

        pos_items = interaction[self.TARGET_ITEM_ID]  # [B]
        neg_items = interaction[self.NEG_ITEM_ID]  # [B] sample 1 negative item

        pos_items_emb = self.item_embedding(pos_items)  # [B H]
        neg_items_emb = self.item_embedding(neg_items)

        pos_bias = self.bias(pos_items)  # [B 1]
        neg_bias = self.bias(neg_items)

        pos_score = pos_bias - self.l2_distance(output, pos_items_emb)
        neg_score = neg_bias - self.l2_distance(output, neg_items_emb)

        bpr_loss = self.bpr_loss(pos_score, neg_score)

        item_emb_loss = self.emb_loss(self.item_embedding.weight)
        user_emb_loss = self.emb_loss(self.user_embedding.weight)
        bias_emb_loss = self.emb_loss(self.bias.weight)
        reg_loss = self.reg_loss(self.T)
        return bpr_loss + item_emb_loss + user_emb_loss + bias_emb_loss + reg_loss

    # TODO implemented after the data interface is ready
    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        output = self.forward(interaction) # [user_num H]

        test_item_emb = self.item_embedding.weight # [item_num H]
        test_item_emb = test_item_emb.repeat(output.size(0), 1, 1) # [user_num item_num H]

        user_hidden = output.unsqueeze(1).expand_as(test_item_emb) # [user_num item_num H]
        test_bias = self.bias.weight # [item_num 1]
        test_bias = test_bias.repeat(user_hidden.size(0), 1, 1) # [user_num item_num 1]

        scores = test_bias - self.l2_distance(user_hidden, test_item_emb) # [user_num item_num 1]
        scores = scores.squeeze(-1)
        return scores
