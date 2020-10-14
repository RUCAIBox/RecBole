# -*- coding: utf-8 -*-
# @Time   : 2020/9/21
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmai.com

# UPDATE
# @Time   : 2020/10/6
# @Author : Zhichao Feng
# @email  : fzcbupt@gmai.com

r"""
recbox.model.context_aware_recommender.din
##############################################
Reference:
Guorui Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" in ACM SIGKDD 2018

Reference code:
https://github.com/zhougr1993/DeepInterestNetwork/tree/master/din
https://github.com/shenweichen/DeepCTR-Torch/tree/master/deepctr_torch/models

"""

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from recbox.utils import ModelType, InputType, FeatureType
from recbox.model.layers import FMEmbedding, MLPLayers, SequenceAttLayer, ContextSeqEmbLayer
from recbox.model.abstract_recommender import SequentialRecommender

class DIN(SequentialRecommender):
    """Deep Interest Network utilizes the attention mechanism to get the weight of each user's behavior according to the target items,
    and finally gets the user representation.

    Note:
        In the official source code, unlike the paper, user features and context features are not input into DNN.
        We just migrated and changed the official source code.
        But You can get user features embedding from user_feat_list.
        Besides, in order to compare with other models, we use AUC instead of GAUC to evaluate the model.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DIN, self).__init__()

        # get field names and parameter value from config
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.LABEL_FIELD = config['LABEL_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.device = config['device']
        self.embedding_size = config['embedding_size']
        self.max_len = config['MAX_ITEM_LIST_LENGTH']
        self.dropout = config['dropout']
        # self.dataset = dataset

        self.types = ['user', 'item']
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        # init MLP layers
        # self.dnn_list = [(3 * self.num_feature_field['item'] + self.num_feature_field['user'])
        #                  * self.embedding_size] + self.mlp_hidden_size
        num_item_feature = len(self.item_feat.interaction.keys())
        self.dnn_list = [
            (3 * num_item_feature) * self.embedding_size
        ] + self.mlp_hidden_size
        self.att_list = [
            4 * num_item_feature * self.embedding_size
        ] + self.mlp_hidden_size

        mask_mat = torch.arange(self.max_len).to(self.device).view(
            1, -1)  # init mask
        self.attention = SequenceAttLayer(mask_mat,
                                          self.att_list,
                                          activation='Sigmoid',
                                          softmax_stag=False,
                                          return_seq_weight=False)
        self.dnn_mlp_layers = MLPLayers(self.dnn_list,
                                        activation='Dice',
                                        dropout=self.dropout,
                                        bn=True)

        self.embedding_layer = ContextSeqEmbLayer(dataset, config)
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        item_list = interaction[self.ITEM_ID_LIST]
        user_list = interaction[self.USER_ID]
        item_list_len = interaction[self.ITEM_LIST_LEN]
        target_item_list = interaction[self.TARGET_ITEM_ID]
        max_length = item_list.shape[1]

        # concatenate the history item list with the target item to get embedding together
        item_target_list = torch.cat((item_list, target_item_list.unsqueeze(1)), dim=-1)

        sparse_embedding, dense_embedding = self.embedding_layer(user_list, item_target_list)

        # concat the sparse embedding and float embedding
        feature_table = {}
        for type in self.types:
            feature_table[type] = []
            if sparse_embedding[type] is not None:
                feature_table[type].append(sparse_embedding[type])
            if dense_embedding[type] is not None:
                feature_table[type].append(dense_embedding[type])

            feature_table[type] = torch.cat(feature_table[type], dim=1)
            table_shape = feature_table[type].shape

            feat_num, embedding_size = table_shape[-2], table_shape[-1]
            feature_table[type] = feature_table[type].view(table_shape[:-2] + (feat_num * embedding_size,))

        user_feat_list = feature_table['user']
        item_feat_list, target_item_feat_emb = feature_table['item'].split([max_length, 1], dim=1)
        target_item_feat_emb = target_item_feat_emb.squeeze()

        # attention
        user_emb = self.attention(target_item_feat_emb, item_feat_list, item_list_len)
        user_emb = user_emb.squeeze()

        # input the DNN to get the prediction score
        din_in = torch.cat([user_emb, target_item_feat_emb,
                            user_emb * target_item_feat_emb], dim=-1)
        din_out = self.dnn_mlp_layers(din_in)
        preds = self.dnn_predict_layers(din_out)
        preds = self.sigmoid(preds)

        return preds.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL_FIELD]
        output = self.forward(interaction)
        loss = self.loss(output, label)
        return loss

    # TODO: Merge the two predict functions when the data interface is ready
    def predict(self, interaction):
        scores = self.forward(interaction)
        return scores
