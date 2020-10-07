# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 16:57
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
recbox.model.sequential_recommender.gru4recf
################################################

Reference:
Balázs Hidasi et al. "Parallel Recurrent Neural Network Architectures for
Feature-rich Session-based Recommendations." in RecSys 2016.

"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbox.utils import InputType
from recbox.model.abstract_recommender import SequentialRecommender
from recbox.model.loss import BPRLoss
from recbox.model.init import xavier_normal_initialization

class GRU4RecF(SequentialRecommender):
    r"""
    In the original paper, the authors proposed several architectures. We compared 3 different
    architectures:
    (1)  Concatenate item input and feature input and use single RNN,
    (2)  Concatenate outputs from two different RNNs,
    (3)  Weighted sum of outputs from two different RNNs.

    We implemented the optimal parallel version(2), which uses different RNNs to
    encode items and features respectively and concatenates the two subparts's
    outputs as the final output. The different RNN encoders are trained simultaneously.
    """
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
        self.feature_count = dataset.num(self.FEATURE_FIELD)
        self.item_feat = dataset.get_item_feature()
        print(self.item_feat.interaction.keys())

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

        self.dense_layer = nn.Linear(self.hidden_size*2, self.embedding_size)
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.loss_type = config['loss_type'] # BPR or CE
        self.bpr_loss = BPRLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.apply(xavier_normal_initialization)

    def gather_indexes(self, gru_output, gather_index):
        "Gathers the vectors at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, gru_output.size(-1))
        output_tensor = gru_output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def load_kg_embedding(self):
        "For GRU4Rec+KG"
        pass

    def forward(self, interaction):
        item_seq = interaction[self.ITEM_ID_LIST]
        feature_seq = self.item_feat[self.FEATURE_FIELD][item_seq].to(item_seq.device)

        # 1. shape [B Len num] means the item has multi-feature, i.e. one movie may be classified
        # into multi-class. We would use sum of the features as the input.

        # 2. shape [B Len] means the item has single-feature, i.e. one store could only in one city.

        item_emb = self.item_embedding(item_seq)
        feature_emb = self.feature_embedding(feature_seq)

        if feature_seq.dim() == 3: # pos_features [B Len Num]
            feature_mask = (feature_seq != 0).float()
            # set the padding as zero
            feature_mask = feature_mask.unsqueeze(-1).expand_as(feature_emb)
            feature_emb = (feature_emb * feature_mask).sum(dim=-2) # [B Len H]


        item_emb = self.dropout(item_emb)
        feature_emb = self.dropout(feature_emb)

        item_gru_output, _ = self.item_gru_layers(item_emb) # [B Len H]
        feature_gru_output, _ = self.feature_gru_layers(feature_emb) # [B Len H]

        output_concat = torch.cat((item_gru_output, feature_gru_output), -1)  # [B Len 2*H]
        output = self.dense_layer(output_concat)
        output = self.gather_indexes(output, interaction[self.ITEM_LIST_LEN] - 1)  # [B H]
        return output # [B H]

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction)
        pos_items = interaction[self.TARGET_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items) # [B H]
            neg_items_emb = self.item_embedding(neg_items) # [B H]
            pos_score = torch.sum(seq_output*pos_items_emb, dim=-1) # [B]
            neg_score = torch.sum(seq_output*neg_items_emb, dim=-1) # [B]
            loss = self.bpr_loss(pos_score, neg_score)
            return loss
        elif self.loss_type == 'CE':
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.ce_loss(logits, pos_items)
            return loss
        else:
            raise NotImplementedError

    # TODO implemented after the data interface is ready
    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # [B, item_num]
        return scores