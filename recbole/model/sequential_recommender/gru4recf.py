# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 16:57
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
GRU4RecF
################################################

Reference:
    Balázs Hidasi et al. "Parallel Recurrent Neural Network Architectures for
    Feature-rich Session-based Recommendations." in RecSys 2016.

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.init import xavier_normal_initialization


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

    def __init__(self, config, dataset):
        super(GRU4RecF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])

        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.feature_embed_layer = FeatureSeqEmbLayer(dataset, self.embedding_size, self.selected_features,
                                                      self.pooling_mode, self.device)
        self.item_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        # For simplicity, we use same architecture for item_gru and feature_gru
        self.feature_gru_layers = nn.GRU(
            input_size=self.embedding_size * self.num_feature_field,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense_layer = nn.Linear(self.hidden_size * 2, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.dropout(item_seq_emb)
        item_gru_output, _ = self.item_gru_layers(item_seq_emb_dropout)  # [B Len H]

        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        feature_table = torch.cat(feature_table, dim=-2)
        # [batch len num_features hidden_size]
        table_shape = feature_table.shape

        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(table_shape[:-2] + (feat_num * embedding_size,))
        feature_gru_output, _ = self.feature_gru_layers(feature_emb) # [B Len H]

        output_concat = torch.cat((item_gru_output, feature_gru_output), -1)  # [B Len 2*H]
        output = self.dense_layer(output_concat)
        output = self.gather_indexes(output, item_seq_len - 1)  # [B H]
        return output # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items) # [B H]
            neg_items_emb = self.item_embedding(neg_items) # [B H]
            pos_score = torch.sum(seq_output*pos_items_emb, dim=-1) # [B]
            neg_score = torch.sum(seq_output*neg_items_emb, dim=-1) # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
