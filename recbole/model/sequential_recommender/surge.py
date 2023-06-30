# -*- coding: utf-8 -*-
# @Time   : 2023/6/30
# @Author : Zhengle Wang
# @Email  : wzl1224413636@gmail.com

# @Time   : 2023/6/30
# @Author : Zhengle Wang
# @Email  : wzl1224413636@gmail.com

import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from recbole.utils import InputType, FeatureType
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.sequential_recommender.dien import AUGRUCell, DynamicRNN
from recbole.model.layers import MLPLayers, SequenceAttLayer
from recbole.model.loss import BPRLoss


class SURGE(SequentialRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SURGE, self).__init__(config, dataset)

        # load dataset info
        self.LABEL_FIELD = config["LABEL_FIELD"]

        # load parameter info
        self.metric_heads = config['metric_heads']
        self.attention_heads = config['attention_heads']
        self.pool_layers = config['pool_layers']
        self.max_seq_len = config['max_seq_len']
        self.pool_length = config['pool_length']
        self.relative_threshold = config['relative_threshold']
        self.device = config['device']
        self.embedding_size = config["embedding_size"]
        self.att_fcn_layer_sizes = config['att_fcn_layer_sizes']
        self.att_activation = config['att_activation']
        self.hidden_size = config['hidden_size']
        self.layer_sizes = config['layer_sizes']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.att_fcn_layer = MLPLayers([4 * self.embedding_size, *self.att_fcn_layer_sizes],
                                       activation=self.att_activation)
        self.dnn_fcn_layer = MLPLayers([3 * self.embedding_size + 1, *self.layer_sizes, 1],
                                       activation=self.att_activation)
        self.dynamic_rnn = DynamicRNN(input_size=self.embedding_size, hidden_size=self.hidden_size,
                                      gru="AUGRU").to(self.device)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, item_seq, pos_item):
        """ SURGE Model: 
			1) Interest graph: Graph construction based on metric learning
			2) Interest fusion and extraction : Graph convolution and graph pooling 
			3) Prediction: Flatten pooled graph to reduced sequence
		"""
        self.mask = torch.ones_like(item_seq).to(self.device)
        self.mask = torch.where(item_seq > 0, self.mask, item_seq).type(torch.float32)
        self.target_item_embedding = self.item_embedding(pos_item)

        X = self.item_embedding(item_seq)  # [B * L * D]

        # build interest graph
        S = []
        for _ in range(self.metric_heads):
            # weighted cosine similarity
            self.weighted_tensor = torch.rand(1, X.shape[-1]).to(self.device)
            X_fts = X * self.weighted_tensor.unsqueeze(0)
            X_fts = F.normalize(X_fts, p=2, dim=2)
            S_one = torch.matmul(X_fts, X_fts.permute((0, 2, 1)))
            # min-max normalization for mask
            S_min, _ = torch.min(S_one, -1, keepdim=True)
            S_max, _ = torch.max(S_one, -1, keepdim=True)
            S_one = (S_one - S_min) / (S_max - S_min)
            S += [S_one]
        S = torch.mean(torch.stack(S, 0), 0).to(self.device)
        # mask invalid nodes
        S = S * self.mask.unsqueeze(-1) * self.mask.unsqueeze(-2)

        # graph sparsification via seted sparseness
        S_flatten = torch.reshape(S, (S.shape[0], -1))  # [B * L^2]
        S_sorted_flatten, _ = torch.sort(S_flatten, dim=-1, descending=True)

        # relative ranking strategy of the entire graph
        num_edges = torch.count_nonzero(S, dim=(1, 2)).type(torch.float32)  # total number of valid edges
        to_keep_edge = torch.ceil(num_edges * self.relative_threshold).type(torch.int32)
        inds = to_keep_edge.clone().detach().unsqueeze(-1).type(torch.int64)
        threshold_score = torch.gather(S_sorted_flatten, 1, inds)
        A = torch.gt(S, threshold_score.unsqueeze(-1)).type(torch.float32)

        for l in range(self.pool_layers):
            X, A, graph_readout, alphas = self._interest_fusion_extraction(X, A, layer=l)

        # flatten pooled graph to reduced sequence
        output_shape = self.mask.shape
        sorted_mask_index = torch.argsort(self.mask, dim=-1, descending=True, stable=True).type(torch.int64)
        sorted_mask, _ = torch.sort(self.mask, descending=True, dim=-1)
        sorted_mask_index = sorted_mask_index.unsqueeze(-1).repeat(1, 1, self.embedding_size)
        X = torch.gather(X, 1, sorted_mask_index)

        self.mask = sorted_mask
        self.reduced_sequence_length = torch.sum(self.mask, 1)  # [B]

        # cut useless sequence tail per batch
        self.to_max_length = torch.arange(0, torch.max(self.reduced_sequence_length)).to(self.device).type(torch.int64)
        X = torch.gather(
            X, 1,
            self.to_max_length.unsqueeze(0).unsqueeze(-1).repeat(X.shape[0], 1, self.embedding_size)
        )  # [B * L * D] -> [B * l * D]
        self.mask = torch.gather(
            self.mask, 1,
            self.to_max_length.unsqueeze(0).repeat(X.shape[0], 1)
        )  # [B * L] -> [B * l]
        self.reduced_sequence_length = torch.sum(self.mask, dim=1).to('cpu')

        # use cluster score as attention weights in AUGRU
        _, alphas = self._attention_fcn(self.target_item_embedding, X)

        packed_rnn_outputs = pack_padded_sequence(
            X, lengths=self.reduced_sequence_length, batch_first=True, enforce_sorted=False
        )
        packed_att_outputs = pack_padded_sequence(
            alphas.unsqueeze(-1), lengths=self.reduced_sequence_length, batch_first=True, enforce_sorted=False
        )
        final_state = self.dynamic_rnn(packed_rnn_outputs, packed_att_outputs)
        _, final_state = pad_packed_sequence(final_state, batch_first=True, padding_value=0.0, total_length=X.shape[1])
        model_output = torch.cat([
            final_state.unsqueeze(-1).to(self.device), graph_readout, self.target_item_embedding,
            graph_readout * self.target_item_embedding
        ],
                                 dim=1)
        logit = self.dnn_fcn_layer(model_output)

        return logit

    def _attention_fcn(self, query, key_value):
        """Apply attention by fully connected layers.
		Args:
			query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
			key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.

		Returns:
			att_weights (obj):  Attention weights
		"""
        query_size = query.shape[-1]
        boolen_mask = torch.eq(self.mask, torch.ones_like(self.mask))

        attention_mat = torch.randn(key_value.shape[-1], query_size).to(self.device)
        att_inputs = torch.tensordot(key_value, attention_mat, [[2], [0]]).to(self.device)  # [B * L * D]

        if query.dim() != att_inputs.dim():
            hist_len = att_inputs.shape[1]
            queries = query.repeat(1, hist_len)
            queries = queries.view(att_inputs.shape).to(self.device)
        else:
            queries = query
        att_fcn_input = torch.cat([att_inputs, queries, att_inputs - queries, att_inputs * queries],
                                  dim=-1).to(self.device)  # [B * L * 4D]

        dense = nn.Linear(self.att_fcn_layer_sizes[-1], 1).to(self.device)
        att_fcn_output = dense(self.att_fcn_layer(att_fcn_input))
        att_fcn_output = torch.squeeze(att_fcn_output, dim=-1)
        mask_paddings = torch.ones_like(att_fcn_output) * (-(2 ** 32) + 1)
        att_weights = F.softmax(torch.where(boolen_mask, att_fcn_output, mask_paddings), dim=-1)
        output = key_value * att_weights.unsqueeze(-1)
        return output, att_weights

    def _interest_fusion_extraction(self, X, A, layer):
        # cluster embedding
        A_bool = torch.gt(A, 0).type(A.dtype).to(self.device)
        A_bool = A_bool * (torch.ones(A.shape[1], A.shape[1]).to(self.device)) - torch.eye(A.shape[1]).to(
            self.device
        ) + torch.eye(A.shape[1]).to(self.device)
        D = torch.sum(A_bool, dim=-1)  # [B * L]
        D = torch.sqrt(D)[:, None] + torch.finfo(torch.float32).eps  # B * 1 * L
        A = (A_bool / D) / D.permute((0, 2, 1))  # [B * L * L] / [B * 1 * L] / [B * L * 1]
        X_q = torch.matmul(A, torch.matmul(A, X))  # query matrix [B * L * D]

        Xc = []
        for i in range(self.attention_heads):
            # cluster- and query- aware attention
            _, f_1 = self._attention_fcn(X_q, X)
            _, f_2 = self._attention_fcn(self.target_item_embedding, X)

            # graph attentive convoution
            E = A_bool * f_1.unsqueeze(1) + A_bool * f_2.unsqueeze(1).permute(
                (0, 2, 1)
            )  # [B * L * 1] * [B * 1 * L] -> [B * L * L]
            E = nn.LeakyReLU()(E)
            boolen_mask = torch.eq(A_bool, torch.ones_like(A_bool))
            mask_paddings = torch.ones_like(E) * (-(2 ** 32) + 1)
            E = F.softmax(torch.where(boolen_mask, E, mask_paddings), dim=-1)
            Xc_one = torch.matmul(E, X)
            Xc_one = nn.Linear(Xc_one.shape[-1], self.embedding_size, bias=False).to(self.device)(Xc_one)
            Xc_one += X
            Xc += [nn.LeakyReLU()(Xc_one)]
        Xc = torch.mean(torch.stack(Xc, 0), 0)

        # cluster fitness score
        X_q = torch.matmul(A, torch.matmul(A, Xc))  # [B * L * F]
        cluster_score = []
        for i in range(self.attention_heads):
            _, f_1 = self._attention_fcn(X_q, Xc)
            _, f_2 = self._attention_fcn(self.target_item_embedding, Xc)
            cluster_score += [f_1 + f_2]
        cluster_score = torch.mean(torch.stack(cluster_score, 0), 0)
        boolen_mask = torch.eq(self.mask, torch.ones_like(self.mask))
        mask_paddings = torch.ones_like(cluster_score) * (-(2 ** 32) + 1)
        cluster_score = F.softmax(torch.where(boolen_mask, cluster_score, mask_paddings), dim=-1)

        # graph pooling
        num_nodes = torch.sum(self.mask, dim=1)
        boolean_pool = torch.gt(num_nodes, self.pool_length)
        to_keep = torch.where(
            boolean_pool, (
                self.pool_length + (self.item_seq_len - self.pool_length) / self.pool_layers *
                (self.pool_layers - layer - 1)
            ).type(torch.int32), num_nodes
        )
        cluster_score = cluster_score * self.mask.type(torch.float32)
        sorted_score, _ = torch.sort(cluster_score, dim=-1, descending=True)
        target_score = torch.gather(sorted_score, -1,
                                    to_keep.unsqueeze(-1).type(torch.int64)) + torch.finfo(torch.float32).eps
        topk_mask = torch.gt(cluster_score, target_score)
        self.mask = topk_mask.type(torch.int32)
        self.float_mask = self.mask.type(torch.float32)
        self.reduced_sequence_length = torch.sum(self.mask, dim=1)

        # ensure graph connectivity
        E = E * torch.unsqueeze(self.float_mask, -1) * torch.unsqueeze(self.float_mask, -2)
        A = torch.matmul(torch.matmul(E, A_bool), E.permute((0, 2, 1)))

        # graph readout
        graph_readout = torch.sum(Xc * cluster_score.unsqueeze(-1) * self.float_mask.unsqueeze(-1), dim=1)  # [B * F]

        return Xc, A, graph_readout, cluster_score

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL_FIELD]
        item_seq = interaction[self.ITEM_SEQ]
        pos_item = interaction[self.POS_ITEM_ID]
        self.item_seq_len = interaction[self.ITEM_SEQ_LEN]
        logits = self.forward(item_seq, pos_item).squeeze(-1)
        loss = self.loss_fct(logits, label)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        pos_item = interaction[self.POS_ITEM_ID]
        self.item_seq_len = interaction[self.ITEM_SEQ_LEN]
        logits = self.forward(item_seq, pos_item)
        return nn.Sigmoid()(logits)
