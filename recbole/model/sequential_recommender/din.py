# -*- coding: utf-8 -*-
# @Time   : 2020/9/21
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmail.com

# UPDATE
# @Time   : 2020/10/21
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

r"""
DIN
##############################################
Reference:
    Guorui Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" in ACM SIGKDD 2018

Reference code:
    - https://github.com/zhougr1993/DeepInterestNetwork/tree/master/din
    - https://github.com/shenweichen/DeepCTR-Torch/tree/master/deepctr_torch/models

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MLPLayers, SequenceAttLayer, ContextSeqEmbLayer
from recbole.utils import InputType, FeatureType


class DIN(SequentialRecommender):
    """Deep Interest Network utilizes the attention mechanism to get the weight of each user's behavior according
    to the target items, and finally gets the user representation.

    Note:
        In the official source code, unlike the paper, user features and context features are not input into DNN.
        We just migrated and changed the official source code.
        But You can get user features embedding from user_feat_list.
        Besides, in order to compare with other models, we use AUC instead of GAUC to evaluate the model.

    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DIN, self).__init__(config, dataset)

        # get field names and parameter value from config
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.device = config["device"]
        self.pooling_mode = config["pooling_mode"]
        self.dropout_prob = config["dropout_prob"]

        self.types = ["user", "item"]
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        # init MLP layers
        # self.dnn_list = [(3 * self.num_feature_field['item'] + self.num_feature_field['user'])
        #                  * self.embedding_size] + self.mlp_hidden_size
        num_item_feature = sum(
            (
                1
                if dataset.field2type[field]
                not in [FeatureType.FLOAT_SEQ, FeatureType.FLOAT]
                or field in config["numerical_features"]
                else 0
            )
            for field in self.item_feat.interaction.keys()
        )
        self.dnn_list = [
            3 * num_item_feature * self.embedding_size
        ] + self.mlp_hidden_size
        self.att_list = [
            4 * num_item_feature * self.embedding_size
        ] + self.mlp_hidden_size

        mask_mat = (
            torch.arange(self.max_seq_length).to(self.device).view(1, -1)
        )  # init mask
        self.attention = SequenceAttLayer(
            mask_mat,
            self.att_list,
            activation="Sigmoid",
            softmax_stag=False,
            return_seq_weight=False,
        )
        self.dnn_mlp_layers = MLPLayers(
            self.dnn_list, activation="Dice", dropout=self.dropout_prob, bn=True
        )

        self.embedding_layer = ContextSeqEmbLayer(
            dataset, self.embedding_size, self.pooling_mode, self.device
        )
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ["embedding_layer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user, item_seq, item_seq_len, next_items):
        max_length = item_seq.shape[1]
        # concatenate the history item seq with the target item to get embedding together
        item_seq_next_item = torch.cat((item_seq, next_items.unsqueeze(1)), dim=-1)
        sparse_embedding, dense_embedding = self.embedding_layer(
            user, item_seq_next_item
        )
        # concat the sparse embedding and float embedding
        feature_table = {}
        for type in self.types:
            feature_table[type] = []
            if sparse_embedding[type] is not None:
                feature_table[type].append(sparse_embedding[type])
            if dense_embedding[type] is not None:
                feature_table[type].append(dense_embedding[type])

            feature_table[type] = torch.cat(feature_table[type], dim=-2)
            table_shape = feature_table[type].shape
            feat_num, embedding_size = table_shape[-2], table_shape[-1]
            feature_table[type] = feature_table[type].view(
                table_shape[:-2] + (feat_num * embedding_size,)
            )

        user_feat_list = feature_table["user"]
        item_feat_list, target_item_feat_emb = feature_table["item"].split(
            [max_length, 1], dim=1
        )
        target_item_feat_emb = target_item_feat_emb.squeeze(1)

        # attention
        user_emb = self.attention(target_item_feat_emb, item_feat_list, item_seq_len)
        user_emb = user_emb.squeeze(1)

        # input the DNN to get the prediction score
        din_in = torch.cat(
            [user_emb, target_item_feat_emb, user_emb * target_item_feat_emb], dim=-1
        )
        din_out = self.dnn_mlp_layers(din_in)
        preds = self.dnn_predict_layers(din_out)

        return preds.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL_FIELD]
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        output = self.forward(user, item_seq, item_seq_len, next_items)
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        scores = self.sigmoid(self.forward(user, item_seq, item_seq_len, next_items))
        return scores
