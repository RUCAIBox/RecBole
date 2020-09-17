# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 16:57
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
Reference:
Balázs Hidasi et al. "Parallel Recurrent Neural Network Architectures for
Feature-rich Session-based Recommendations." in RecSys 2016.
"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from ...utils import InputType
from ..abstract_recommender import SequentialRecommender
from ..utils import xavier_normal_initialization
from ..loss import BPRLoss

class GRU4RecF(SequentialRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(GRU4RecF, self).__init__()

        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.FEATURE_FIELD = config['FEATURE_FIELD']
        self.FEATURE_LIST = self.FEATURE_FIELD + config['LIST_SUFFIX']

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        self.item_count = dataset.item_num
        self.feature_count = dataset.num('class')
        self.item_feat = dataset.get_item_feature()
        print(self.item_feat.interaction.keys()) # ['item_id' 'class']

        # todo: now only consider movie category

        self.item_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)
        self.feature_embedding = nn.Embedding(self.feature_count, self.embedding_size, padding_idx=0)

        # For simplicity, we use same architecture for item_gru and feature_gru

        self.item_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.feature_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.bpr_loss = BPRLoss()

        self.apply(xavier_normal_initialization)

    def gather_indexes(self, gru_output, gather_index):
        "Gathers the vectors at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, self.embedding_size)
        output_tensor = gru_output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, interaction):
        pos_items = interaction[self.ITEM_ID_LIST]
        pos_features = self.item_feat['class'][pos_items]
        pos_features = pos_features.to(pos_items.device) # [B Len 6] padding zero for genres

        item_emb = self.item_embedding(pos_items)
        feature_emb = self.feature_embedding(pos_features) # [B Len 6 H]

        feature_mask = (pos_features != 0).float()
        feature_mask = feature_mask.unsqueeze(-1).expand_as(feature_emb)
        feature_emb = (feature_emb * feature_mask).sum(dim=-2) # [B Len H]

        item_gru_output, _ = self.item_gru_layers(item_emb) # [B Len H]
        feature_gru_output, _ = self.feature_gru_layers(feature_emb) # [B Len H]

        item_output = self.gather_indexes(item_gru_output, interaction[self.ITEM_LIST_LEN] - 1) # [B H]
        feature_output = self.gather_indexes(feature_gru_output, interaction[self.ITEM_LIST_LEN] - 1) # [B H]
        output = 0.5*item_output + 0.5*feature_output
        return output # [B H]

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction)
        pos_items = interaction[self.TARGET_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_items_emb = self.item_embedding(pos_items) # [B H]
        neg_items_emb = self.item_embedding(neg_items) # [B H]
        pos_score = torch.sum(seq_output*pos_items_emb, dim=-1) # [B]
        neg_score = torch.sum(seq_output*neg_items_emb, dim=-1) # [B]

        loss = self.bpr_loss(pos_score, neg_score)
        return loss

    # todo: need user_item_seq and item to test
    def predict(self, interaction):
        seq_output = self.forward(interaction)
        # todo: get this field
        test_item_emb = self.item_embedding(interaction[self.TARGET_ITEM_ID]) # [B H]
        scores = torch.sum(seq_output*test_item_emb, dim=-1)
        return scores

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # [B, item_num]
        return scores