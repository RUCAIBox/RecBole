# -*- coding: utf-8 -*-
# @Time   : 2020/9/21
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmai.com

'''
Reference:
Guorui Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" in ACM SIGKDD 2018

Note:
    In this implementation, dropout is used instead of mini batch aware regularizat.
    In order to compare with other models, we use AUC instead of GAUC to evaluate the model.
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from recbox.utils import ModelType, InputType, FeatureType
from recbox.model.abstract_recommender import SequentialRecommender
from recbox.model.layers import FMEmbedding
from recbox.model.layers import MLPLayers

class DIN(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DIN, self).__init__()
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.LABEL_FIELD = config['LABEL_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.device = config['device']
        self.types = ['user', 'item']
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()
        self.user_feat = self.user_feat.to(self.device)
        self.item_feat = self.item_feat.to(self.device)
        self.embedding_size = config['embedding_size']
        self.dropout = config['dropout']

        # get user feature field and item feature field
        self.field_names = {'user': list(self.user_feat.interaction.keys()),
                            'item': list(self.item_feat.interaction.keys())}

        self.token_field_names = {type: [] for type in self.types}
        self.token_field_dims = {type: [] for type in self.types}
        self.float_field_names = {type: [] for type in self.types}
        self.float_field_dims = {type: [] for type in self.types}
        self.token_seq_field_names = {type: [] for type in self.types}
        self.token_seq_field_dims = {type: [] for type in self.types}
        self.num_feature_field = {type: 0 for type in self.types}

        for type in self.types:
            for field_name in self.field_names[type]:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.token_field_names[type].append(field_name)
                    self.token_field_dims[type].append(dataset.num(field_name))
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.token_seq_field_names[type].append(field_name)
                    self.token_seq_field_dims[type].append(dataset.num(field_name))
                else:
                    self.float_field_names[type].append(field_name)
                    self.float_field_dims[type].append(dataset.num(field_name))
                self.num_feature_field[type] += 1

        self.token_field_offsets = {}
        self.token_embedding_table = {}
        self.float_embedding_table = {}
        self.token_seq_embedding_table = {}

        # get embedding
        for type in self.types:
            if len(self.token_field_dims[type]) > 0:
                self.token_field_offsets[type] = np.array((0, *np.cumsum(self.token_field_dims[type])[:-1]),
                                                            dtype=np.long)

                self.token_embedding_table[type] = FMEmbedding(self.token_field_dims[type],
                                                                 self.token_field_offsets[type],
                                                                 self.embedding_size).to(self.device)
            if len(self.float_field_dims[type]) > 0:
                self.float_embedding_table[type] = nn.Embedding(np.sum(self.float_field_dims[type], dtype=np.int32),
                                                                  self.embedding_size).to(self.device)
            if len(self.token_seq_field_dims) > 0:
                self.token_seq_embedding_table[type] = nn.ModuleList()
                for token_seq_field_dim in self.token_seq_field_dims[type]:
                    self.token_seq_embedding_table[type].append(
                        nn.Embedding(token_seq_field_dim, self.embedding_size).to(self.device))

        # init MLP layers
        # self.dnn_list = [(3 * self.num_feature_field['item'] + self.num_feature_field['user'])
        #                  * self.embedding_size] + self.mlp_hidden_size

        self.dnn_list = [(3 * self.num_feature_field['item']) * self.embedding_size] + self.mlp_hidden_size

        self.att_list = [4 * self.num_feature_field['item'] * self.embedding_size] + self.mlp_hidden_size
        self.dnn_mlp_layers = MLPLayers(self.dnn_list, activation='Dice', dropout=self.dropout, bn=True)
        self.att_mlp_layers = MLPLayers(self.att_list, activation='Sigmoid', dropout=self.dropout, bn=False)
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.dense = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.max_len = config['MAX_ITEM_LIST_LENGTH']
        self.mask_mat = torch.arange(self.max_len).to(self.device).view(1, -1)

        self.PredictionLayer = PredictionLayer()
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)
        self.loss = nn.BCELoss()

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

        # concatenate the product history sequence with the target product to get embedding together
        item_target_list = torch.cat((item_list, target_item_list.unsqueeze(1)), dim=-1)

        sparse_embedding, dense_embedding = self.embed_input_fields(user_list, item_target_list)
        feature_table = {}

        # concat the sparse embedding and float embedding
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
        preds = self.PredictionLayer(preds)

        return preds.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL_FIELD]
        output = self.forward(interaction)
        loss = self.loss(output, label)
        return loss

    # TODO: Merge the two predict functions when the data interface is ready
    def predict(self, interaction):
        x = self.forward(interaction)
        return x

    # def predict(self, interaction):
    #     item_list = interaction[self.ITEM_ID_LIST]
    #     user_list = interaction[self.USER_ID]
    #     item_list_len = interaction[self.ITEM_LIST_LEN]
    #     target_item_list = interaction['item_id']
    #     max_length = item_list.shape[1]
    #
    #     item_target_list = torch.cat((item_list, target_item_list.unsqueeze(1)), dim=-1)
    #
    #     sparse_embedding, dense_embedding = self.embed_input_fields(user_list, item_target_list)
    #
    #     feature_table = {}
    #     for type in self.types:
    #         feature_table[type] = []
    #         if sparse_embedding[type] is not None:
    #             feature_table[type].append(sparse_embedding[type])
    #         if dense_embedding[type] is not None:
    #             feature_table[type].append(dense_embedding[type])
    #
    #         feature_table[type] = torch.cat(feature_table[type], dim=1)
    #         table_shape = feature_table[type].shape
    #         feat_num, embedding_size = table_shape[-2], table_shape[-1]
    #         feature_table[type] = feature_table[type].view(table_shape[:-2] + (feat_num * embedding_size,))
    #
    #     user_feat_list = feature_table['user']
    #     item_feat_list, target_item_feat_emb = feature_table['item'].split([max_length, 1], dim=1)
    #     target_item_feat_emb = target_item_feat_emb.squeeze()
    #
    #     user_emb = self.attention(target_item_feat_emb, item_feat_list, item_list_len)
    #     user_emb = user_emb.squeeze()
    #
    #     din_in = torch.cat([user_emb, target_item_feat_emb,
    #                         user_emb * target_item_feat_emb], dim=-1)
    #     din_out = self.dnn_mlp_layers(din_in)
    #     preds = self.dnn_predict_layers(din_out)
    #     preds = self.PredictionLayer(preds)
    #
    #     return preds.squeeze(1)

    '''
        In the following three functions, when the type is user, [batch_ size, max_ item_length] should be changed
        to [batch_ size]
    '''

    def embed_float_fields(self, float_fields, type, embed=True):
        '''
        :param float_fields: [batch_size, max_item_length, num_float_field]
        :param type: user or item
        :param embed: bool
        :return: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]
        '''
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[1]
        # [batch_size, max_item_length, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, max_item_length, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table[type](index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        return float_embedding

    def embed_token_fields(self, token_fields, type):
        '''
        :param token_fields: [batch_size, max_item_length, num_token_field]
        :param type: user or item
        :return: token fields embedding. [batch_size, max_item_length, num_token_field, embed_dim]
        '''
        if token_fields is None:
            return None
        # [batch_size, max_item_length, num_token_field, embed_dim]
        if type == 'item':
            token_fields = token_fields.transpose(-1, -2)
            embedding_shape = token_fields.shape + (-1,)
            token_fields = token_fields.reshape(-1, token_fields.shape[-1])
            token_embedding = self.token_embedding_table[type](token_fields)
            token_embedding = token_embedding.view(embedding_shape)
        else:
            token_embedding = self.token_embedding_table[type](token_fields)
        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, type, mode='mean'):
        '''
        :param token_seq_fields: [batch_size, max_item_length, seq_len]`
        :param type: user or item
        :param mode: mean/max/sum
        :return: [batch_size, max_item_length, num_token_seq_field, embed_dim]
        '''
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[type][i]
            mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=-1, keepdim=True)  # [batch_size, max_item_length, 1]
            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, max_item_length, seq_len, embed_dim]
            mask = mask.unsqueeze(-1).expand_as(
                token_seq_embedding)  # [batch_size, max_item_length, seq_len, embed_dim]
            if mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (
                        1 - mask) * 1e9  # [batch_size, max_item_length, seq_len, embed_dim]
                result = torch.max(masked_token_seq_embedding, dim=-2,
                                   keepdim=True)  # [batch_size, max_item_length, 1, embed_dim]
                # result = result.values
            elif mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=-2,
                                   keepdim=True)  # [batch_size, max_item_length, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=-2)  # [batch_size, max_item_length, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, max_item_length, embed_dim]
                result = result.unsqueeze(-2)  # [batch_size, max_item_length, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=-2)  # [batch_size, max_item_length, num_token_seq_field, embed_dim]

    # get embedding of user_idx and item_idx
    def embed_input_fields(self, user_idx, item_idx):
        '''
        :param user_idx: interaction['user_id']
        :param item_idx: interaction['item_id_list']
        :return: dict
        '''
        user_item_feat = {'user': self.user_feat, 'item': self.item_feat}
        user_item_idx = {'user': user_idx, 'item': item_idx}
        float_fields_embedding = {}
        token_fields_embedding = {}
        token_seq_fields_embedding = {}
        sparse_embedding = {}
        dense_embedding = {}

        for type in self.types:
            float_fields = []
            for field_name in self.float_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_fields.append(feature
                                    if len(feature.shape) == 2
                                    else feature.unsqueeze(1))
            if len(float_fields) > 0:
                float_fields = torch.cat(float_fields, dim=1)  # [batch_size, max_item_length, num_float_field]
            else:
                float_fields = None
            # [batch_size, max_item_length, num_float_field]
            # or [batch_size, max_item_length, num_float_field, embed_dim] or None
            float_fields_embedding[type] = self.embed_float_fields(float_fields, type)

            token_fields = []
            for field_name in self.token_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_fields.append(feature.unsqueeze(1))
            if len(token_fields) > 0:
                token_fields = torch.cat(token_fields, dim=1)  # [batch_size, max_item_length, num_token_field]
            else:
                token_fields = None
            # [batch_size, max_item_length, num_token_field, embed_dim] or None
            token_fields_embedding[type] = self.embed_token_fields(token_fields, type)

            token_seq_fields = []
            for field_name in self.token_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_seq_fields.append(feature)
            # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
            token_seq_fields_embedding[type] = self.embed_token_seq_fields(token_seq_fields, type)

            if token_fields_embedding[type] is None:
                sparse_embedding[type] = token_seq_fields_embedding[type]
            else:
                if token_seq_fields_embedding[type] is None:
                    sparse_embedding[type] = token_fields_embedding[type]
                else:
                    sparse_embedding[type] = torch.cat([token_fields_embedding[type],
                                                          token_seq_fields_embedding[type]], dim=-2)
            dense_embedding[type] = float_fields_embedding[type]

            # sparse_embedding[type] shape: [batch_size, max_item_length, num_token_seq_field+num_token_field, embed_dim] or None
            # dense_embedding[type] shape: [batch_size, max_item_length, num_float_field] or [batch_size, max_item_length, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding

    def attention(self, queries, keys, keys_length):
        """
        :param queries:       [B, H] candidate ads
        :param keys:          [B, T, H] user_hist
        :param keys_length:   [B] mask

        H means embedding_size * feat_num
        """
        embbedding_size = queries.shape[-1]  # H
        hist_len = keys.shape[1]  # T
        queries = queries.repeat(1, hist_len)

        queries = queries.view(-1, hist_len, embbedding_size)

        # MLP Layer
        input = torch.cat(
            [queries, keys, queries - keys, queries * keys], dim=-1)
        output = self.att_mlp_layers(input)
        output = torch.transpose(self.dense(output), -1, -2)

        # get mask
        output = output.squeeze(1)
        mask = self.mask_mat.repeat(output.size(0), 1)
        mask = (mask >= keys_length.unsqueeze(1))
        # mask
        output = output.masked_fill(mask=mask, value=torch.tensor(-np.inf))
        output = output.unsqueeze(1)
        output = output / (embbedding_size ** 0.5)

        # Get the weight of each user's history list about the target item
        output = F.softmax(output, dim=2)  # [B, 1, T]
        output = torch.matmul(output, keys)  # [B, 1, H]

        return output


class PredictionLayer(nn.Module):

    def __init__(self):
        super(PredictionLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, preds):
        output = preds + self.bias
        output = torch.sigmoid(output)

        return output
