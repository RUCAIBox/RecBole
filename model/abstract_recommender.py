# -*- coding: utf-8 -*-
# @Time   : 2020/6/25 15:47
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : abstract_recommender.py

import numpy as np
import torch
import torch.nn as nn
from utils import ModelType
from model.layers import FMEmbedding, FMFirstOrderLinear


class AbstractRecommender(nn.Module):
    """
    Base class for all models
    """

    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def calculate_loss(self, *inputs):
        """
         Calculate Train loss

        :return: Model train loss
        """
        raise NotImplementedError

    def predict(self, *inputs):
        """
         Result prediction for testing and evaluating

        :return: Model predict
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    def __init__(self):
        super(GeneralRecommender, self).__init__()
        self.type = ModelType.GENERAL


class SequentialRecommender(AbstractRecommender):
    def __init__(self):
        super(SequentialRecommender, self).__init__()
        self.type = ModelType.SEQUENTIAL


class ContextRecommender(AbstractRecommender):
    def __init__(self, config, dataset):
        super(ContextRecommender, self).__init__()
        self.type = ModelType.CONTEXT

        self.field_names = dataset.fields()
        self.LABEL = config['LABEL_FIELD']
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue

            if dataset.field2type[field_name] == 'token':
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == 'token_seq':
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))

        self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
        self.token_embedding_table = FMEmbedding(self.token_field_dims, self.token_field_offsets, self.embedding_size)
        self.float_embedding_table = nn.Embedding(np.sum(self.float_field_dims, axis=None), self.embedding_size)
        self.token_seq_embedding_table = nn.ModuleList()
        for token_seq_field_dim in self.token_seq_field_dims:
            self.token_seq_embedding_table.append(nn.Embedding(token_seq_field_dim, self.embedding_size))

        self.first_order_linear = FMFirstOrderLinear(config, dataset)

    def embed_float_fields(self, float_fields, embed=True):
        # input Tensor shape : [batch_size, num_float_field]
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[1]
        # [batch_size, num_float_field]
        index = torch.arange(1, num_float_field).unsqueeze(0).expand_as(float_fields).long()

        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        return float_embedding

    def embed_token_fields(self, token_fields):
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode='mean'):
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(token_seq_embedding)  # [batch_size, seq_len, embed_dim]
            if mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]
                result = torch.max(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            elif mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1)  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=1)  # [batch_size, num_token_seq_field, embed_dim]

    def embed_input_fields(self, interaction):
        float_fields = []
        for field_name in self.float_field_names:
            float_fields.append(interaction[field_name]
                                if len(interaction[field_name].shape) == 2 else interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)  # [batch_size, num_float_field]
        else:
            float_fields = None
        # [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat([token_fields_embedding, token_seq_fields_embedding], dim=1)

        dense_embedding = float_fields_embedding

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding
