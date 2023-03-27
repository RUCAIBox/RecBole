# -*- coding: utf-8 -*-
# @Time     : 2020/11/22 14:56
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun
# @Email    : shaoweiqi@ruc.edu.cn

r"""
NPE
################################################

Reference:
    ThaiBinh Nguyen, et al. "NPE: Neural Personalized Embedding for Collaborative Filtering" in IJCAI 2018.

Reference code:
    https://github.com/wubinzzu/NeuRec

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class NPE(SequentialRecommender):
    r"""
    models a user’s click to an item in two terms: the personal preference of the user for the item,
    and the relationships between this item and other items clicked by the user

    """

    def __init__(self, config, dataset):
        super(NPE, self).__init__(config, dataset)

        # load the dataset information
        self.n_user = dataset.num(self.USER_ID)
        self.device = config["device"]

        # load the parameters information
        self.embedding_size = config["embedding_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss type
        self.user_embedding = nn.Embedding(self.n_user, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.embedding_seq_item = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)

        self.loss_type = config["loss_type"]
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameters of the module
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, seq_item, user):
        user_embedding = self.dropout(self.relu(self.user_embedding(user)))
        # batch_size * embedding_size
        seq_item_embedding = self.item_embedding(seq_item).sum(dim=1)
        seq_item_embedding = self.dropout(self.relu(seq_item_embedding))
        # batch_size * embedding_size

        return user_embedding + seq_item_embedding

    def calculate_loss(self, interaction):
        seq_item = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        seq_output = self.forward(seq_item, user)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_embs = self.item_embedding(pos_items)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.relu(self.item_embedding(neg_items))
            pos_items_emb = self.relu(pos_items_embs)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.relu(self.item_embedding.weight)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user)
        test_item_emb = self.relu(self.item_embedding(test_item))
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user)
        test_items_emb = self.relu(self.item_embedding.weight)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
