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
import numpy as np

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbox.utils import InputType, FeatureType
from recbox.model.abstract_recommender import SequentialRecommender
from recbox.model.loss import BPRLoss
from recbox.model.layers import FMEmbedding
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
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        self.item_count = dataset.item_num
        # list
        self.selected_features = config['selected_features']
        self.dataset = dataset
        self.device = config['device']
        self.item_feat = dataset.get_item_feature().to(self.device)

        # need change the 'load_col' config

        self.hidden_size = config['hidden_size']
        self.embedding_size = config['embedding_size']

        self.get_feat_dims()
        self.get_feat_embeddings()

        self.item_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)


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

        self.dense_layer = nn.Linear(config['hidden_size'] * (1 + self.num_feature_field), self.embedding_size)
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

    def get_feat_dims(self):
        """get user feature field and item feature field.

        """
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.num_feature_field = 0

        for field_name in self.selected_features:
            if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(self.dataset.num(field_name))
            elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(self.dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(self.dataset.num(field_name))
            self.num_feature_field += 1

    def get_feat_embeddings(self):
        """get embedding of all features.

        """
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]),
                                                      dtype=np.long)

            self.token_embedding_table = FMEmbedding(self.token_field_dims,
                                                           self.token_field_offsets,
                                                           self.embedding_size)
        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(np.sum(self.float_field_dims, dtype=np.int32),
                                                      self.embedding_size)
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, self.embedding_size))

    def embed_float_fields(self, float_fields, embed=True):
        """Get the embedding of float fields.
        In the following three functions("embed_float_fields" "embed_token_fields" "embed_token_seq_fields")
        when the type is user, [batch_ size, max_item_length] should be recognised as [batch_size]

        Args:
            float_fields(torch.Tensor): [batch_size, max_item_length, num_float_field]
            embed(bool): embed or not

        Returns:
            torch.Tensor: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]

        """
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[1]
        # [batch_size, max_item_length, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, max_item_length, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        return float_embedding

    def embed_token_fields(self, token_fields):
        """Get the embedding of toekn fields

        Args:
            token_fields(torch.Tensor): input, [batch_size, max_item_length, num_token_field]
            type(str): user or item

        Returns:
            torch.Tensor: token fields embedding, [batch_size, max_item_length, num_token_field, embed_dim]

        """
        if token_fields is None:
            return None
        # [batch_size, max_item_length, num_token_field, embed_dim]
        token_fields = token_fields.transpose(-1, -2)
        embedding_shape = token_fields.shape + (-1,)
        token_fields = token_fields.reshape(-1, token_fields.shape[-1])
        token_embedding = self.token_embedding_table(token_fields)
        token_embedding = token_embedding.view(embedding_shape)
        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode='mean'):
        """Get the embedding of token_seq fields.

        Args:
            token_seq_fields(torch.Tensor): input, [batch_size, max_item_length, seq_len]`
            mode(str): mean/max/sum

        Returns:
            torch.Tensor: result [batch_size, max_item_length, num_token_seq_field, embed_dim]

        """
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
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

    def embed_input_fields(self, item_seq):
        """Get the embeddings of item features

        Args:
            item_idx(torch.Tensor): interaction['item_id']

        Returns:
            dict: embeddings of item feature

        """
        float_fields = []
        for field_name in self.float_field_names:
            feature = self.item_feat[field_name][item_seq]
            float_fields.append(feature
                                if len(feature.shape) == 2
                                else feature.unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)  # [batch_size, max_item_length, num_float_field]
        else:
            float_fields = None
        # [batch_size, max_item_length, num_float_field]
        # or [batch_size, max_item_length, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        token_fields = []
        for field_name in self.token_field_names:
            feature = self.item_feat[field_name][item_seq]
            token_fields.append(feature.unsqueeze(1))

        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)  # [batch_size, max_item_length, num_token_field]
        else:
            token_fields = None
        # [batch_size, max_item_length, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            feature = self.item_feat[field_name][item_seq]
            token_seq_fields.append(feature)
        # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat([token_fields_embedding,
                                            token_seq_fields_embedding], dim=-2)

        dense_embedding = float_fields_embedding

        # sparse_embedding[type] shape: [batch_size, max_item_length, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding[type] shape: [batch_size, max_item_length, num_float_field] or [batch_size, max_item_length, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding


    def forward(self, interaction):
        item_seq = interaction[self.ITEM_ID_LIST]
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_gru_output, _ = self.item_gru_layers(item_emb)  # [B Len H]

        sparse_embedding, dense_embedding = self.embed_input_fields(item_seq)
        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        feature_table = torch.cat(feature_table, dim=1)
        # [batch len num_features hidden_size]
        table_shape = feature_table.shape

        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(table_shape[:-2] + (feat_num * embedding_size,))
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