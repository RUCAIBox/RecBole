# -*- coding: utf-8 -*-
# @Time   : 2020/6/28 15:55
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : gru4rec.py

"""
Reference:
Balázs Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks." in ICLR 2016.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract_recommender import AbstractRecommender


class GRU4Rec(AbstractRecommender):
    def __init__(self, config, dataset):
        super(GRU4Rec, self).__init__()

        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_items = len(dataset.field2id_token[self.ITEM_ID])
        self.embedding_size = config['embedding_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.seq_len = config['seq_len']

        # todo: padding index 0
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.gru_layers = nn.GRU(self.embedding_size, self.embedding_size,
                                 self.num_layers, bias=False, batch_first=True, dropout=self.dropout)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.GRU)):
            xavier_normal_(module.weight)

    def get_item_embedding(self, item):
        return self.item_embedding[item]

    def get_history_embedding(self, history_seq):
        output, hn = self.gru_layers(self.item_embedding(history_seq))
        return output

    def forward(self, history_seq, item):
        history_embedding = self.get_history_embedding(history_seq)
        item_embedding = self.get_item_embedding(item)
        return history_embedding, item_embedding

    def calculate_loss(self, interaction):
        history_seq = interaction['history_seq']
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        seq_emb, pos_emb = self.forward(history_seq, pos_item)
        neg_emb = self.get_history_embedding(neg_item)
        seq_emb = seq_emb.view(-1, self.embedding_size)     # (batch * seq_len, hidden_size)
        pos_emb, neg_emb = pos_emb.view(-1, pos_emb.size(2)), neg_emb.view(-1, neg_emb.size(2))
        pos_logits, neg_logits = torch.sum(pos_emb * seq_emb, -1), torch.sum(neg_emb * seq_emb, -1)  # (batch * seq_len)
        istarget = (pos_item > 0).view(pos_item.size(0) * self.seq_len).float()  # [batch * seq_len]
        loss = torch.sum(
            - torch.log(self.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - self.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        return loss

    def predict(self, interaction):
        history_seq = interaction['history_seq']
        item = interaction[self.ITEM_ID]
        seq_emb, item_emb = self.forward(history_seq, item)
        scores = torch.matmul(seq_emb, item_emb.transpos(1, 2))     # (batch, seq_len, 1 + num_neg_sample)
        scores = scores[:, -1, :]   # (batch, 1 + num_neg_sample)
        return scores
