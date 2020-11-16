# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 17:01
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
TransRec
################################################

Reference:
    Ruining He et al. "Translation-based Recommendation." In RecSys 2017.

"""

import torch
from torch import nn

from recbole.utils import InputType
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss, EmbLoss, RegLoss
from recbole.model.init import xavier_normal_initialization


class TransRec(SequentialRecommender):
    r"""
    TransRec is translation-based model for sequential recommendation.
    It assumes that the `prev. item` + `user`  = `next item`.
    We use the Euclidean Distance to calculate the similarity in this implementation.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(TransRec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']

        # load dataset info
        self.n_users = dataset.user_num

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.bias = nn.Embedding(self.n_items, 1, padding_idx=0) # Beta popularity bias
        self.T = nn.Parameter(torch.zeros(self.embedding_size)) # average user representation 'global'

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_loss = RegLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _l2_distance(self, x, y):
        return torch.sqrt(torch.sum((x - y)**2, dim=-1, keepdim=True))  # [B 1]

    def gather_last_items(self, item_seq, gather_index):
        "Gathers the last_item at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1)
        last_items = item_seq.gather(index=gather_index, dim=1) # [B 1]
        return last_items.squeeze(-1) # [B]

    def forward(self, user, item_seq, item_seq_len):
        # the last item at the last position
        last_items = self.gather_last_items(item_seq, item_seq_len - 1) # [B]
        user_emb = self.user_embedding(user) # [B H]
        last_items_emb = self.item_embedding(last_items)  # [B H]
        T = self.T.expand_as(user_emb) # [B H]
        seq_output = user_emb + T + last_items_emb # [B H]
        return seq_output

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]

        pos_items = interaction[self.POS_ITEM_ID]  # [B]
        neg_items = interaction[self.NEG_ITEM_ID]  # [B] sample 1 negative item

        pos_items_emb = self.item_embedding(pos_items)  # [B H]
        neg_items_emb = self.item_embedding(neg_items)

        pos_bias = self.bias(pos_items)  # [B 1]
        neg_bias = self.bias(neg_items)

        pos_score = pos_bias - self._l2_distance(seq_output, pos_items_emb)
        neg_score = neg_bias - self._l2_distance(seq_output, neg_items_emb)

        bpr_loss = self.bpr_loss(pos_score, neg_score)
        item_emb_loss = self.emb_loss(self.item_embedding(pos_items).detach())
        user_emb_loss = self.emb_loss(self.user_embedding(user).detach())
        bias_emb_loss = self.emb_loss(self.bias(pos_items).detach())

        reg_loss = self.reg_loss(self.T)
        return bpr_loss + item_emb_loss + user_emb_loss + bias_emb_loss + reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding(test_item)  # [B H]
        test_bias = self.bias(test_item)  # [B 1]

        scores = test_bias - self._l2_distance(seq_output, test_item_emb)  # [B 1]
        scores = scores.squeeze(-1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]

        test_items_emb = self.item_embedding.weight # [item_num H]
        test_items_emb = test_items_emb.repeat(seq_output.size(0), 1, 1) # [user_num item_num H]

        user_hidden = seq_output.unsqueeze(1).expand_as(test_items_emb) # [user_num item_num H]
        test_bias = self.bias.weight # [item_num 1]
        test_bias = test_bias.repeat(user_hidden.size(0), 1, 1) # [user_num item_num 1]

        scores = test_bias - self._l2_distance(user_hidden, test_items_emb) # [user_num item_num 1]
        scores = scores.squeeze(-1)  # [B n_items]
        return scores
