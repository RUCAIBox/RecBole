# -*- coding: utf-8 -*-
# @Time   : 2021/2/15
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmail.com

# UPDATE
# @Time   : 2021/5/6
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

r"""
DIEN
##############################################
Reference:
    Guorui Zhou et al. "Deep Interest Evolution Network for Click-Through Rate Prediction" in AAAI 2019

Reference code:
    - https://github.com/mouna99/dien
    - https://github.com/shenweichen/DeepCTR-Torch/

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from recbole.utils import ModelType, InputType, FeatureType
from recbole.model.layers import (
    FMEmbedding,
    MLPLayers,
    ContextSeqEmbLayer,
    SequenceAttLayer,
)
from recbole.model.abstract_recommender import SequentialRecommender


class DIEN(SequentialRecommender):
    """DIEN has an interest extractor layer to capture temporal interests from history behavior sequence,and an
    interest evolving layer to capture interest evolving process that is relative to the target item. At interest
    evolving layer, attention mechanism is embedded intothe sequential structure novelly, and the effects of relative
    interests are strengthened during interest evolution.

    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DIEN, self).__init__(config, dataset)

        # get field names and parameter value from config
        self.device = config["device"]
        self.alpha = config["alpha"]
        self.gru = config["gru_type"]
        self.pooling_mode = config["pooling_mode"]
        self.dropout_prob = config["dropout_prob"]
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.NEG_ITEM_SEQ = config["NEG_PREFIX"] + self.ITEM_SEQ

        self.types = ["user", "item"]
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

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
        num_user_feature = sum(
            (
                1
                if dataset.field2type[field]
                not in [FeatureType.FLOAT_SEQ, FeatureType.FLOAT]
                or field in config["numerical_features"]
                else 0
            )
            for field in self.user_feat.interaction.keys()
        )
        item_feat_dim = num_item_feature * self.embedding_size
        mask_mat = (
            torch.arange(self.max_seq_length).to(self.device).view(1, -1)
        )  # init mask

        # init sizes of used layers
        self.att_list = [
            4 * num_item_feature * self.embedding_size
        ] + self.mlp_hidden_size
        self.interest_mlp_list = [2 * item_feat_dim] + self.mlp_hidden_size + [1]
        self.dnn_mlp_list = [
            2 * item_feat_dim + num_user_feature * self.embedding_size
        ] + self.mlp_hidden_size

        # init interest extractor layer, interest evolving layer embedding layer, MLP layer and linear layer
        self.interset_extractor = InterestExtractorNetwork(
            item_feat_dim, item_feat_dim, self.interest_mlp_list
        )
        self.interest_evolution = InterestEvolvingLayer(
            mask_mat, item_feat_dim, item_feat_dim, self.att_list, gru=self.gru
        )
        self.embedding_layer = ContextSeqEmbLayer(
            dataset, self.embedding_size, self.pooling_mode, self.device
        )
        self.dnn_mlp_layers = MLPLayers(
            self.dnn_mlp_list, activation="Dice", dropout=self.dropout_prob, bn=True
        )
        self.dnn_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)
        self.other_parameter_name = ["embedding_layer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user, item_seq, neg_item_seq, item_seq_len, next_items):
        max_length = item_seq.shape[1]
        # concatenate the history item seq with the target item to get embedding together
        item_seq_next_item = torch.cat(
            (item_seq, neg_item_seq, next_items.unsqueeze(1)), dim=-1
        )
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
        item_feat_list, neg_item_feat_list, target_item_feat_emb = feature_table[
            "item"
        ].split([max_length, max_length, 1], dim=1)
        target_item_feat_emb = target_item_feat_emb.squeeze(1)

        # interest
        interest, aux_loss = self.interset_extractor(
            item_feat_list, item_seq_len, neg_item_feat_list
        )
        evolution = self.interest_evolution(
            target_item_feat_emb, interest, item_seq_len
        )

        dien_in = torch.cat([evolution, target_item_feat_emb, user_feat_list], dim=-1)
        # input the DNN to get the prediction score
        dien_out = self.dnn_mlp_layers(dien_in)
        preds = self.dnn_predict_layer(dien_out)
        return preds.squeeze(1), aux_loss

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL_FIELD]
        item_seq = interaction[self.ITEM_SEQ]
        neg_item_seq = interaction[self.NEG_ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        output, aux_loss = self.forward(
            user, item_seq, neg_item_seq, item_seq_len, next_items
        )
        loss = self.loss(output, label) + self.alpha * aux_loss
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        neg_item_seq = interaction[self.NEG_ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        scores, _ = self.forward(user, item_seq, neg_item_seq, item_seq_len, next_items)
        return self.sigmoid(scores)


class InterestExtractorNetwork(nn.Module):
    """In e-commerce system, user behavior is the carrier of latent interest, and interest will change after
    user takes one behavior. At the interest extractor layer, DIEN extracts series of interest states from
    sequential user behaviors.
    """

    def __init__(self, input_size, hidden_size, mlp_size):
        super(InterestExtractorNetwork, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.auxiliary_net = MLPLayers(layers=mlp_size, activation="none")

    def forward(self, keys, keys_length, neg_keys=None):
        batch_size, hist_len, embedding_size = keys.shape
        packed_keys = pack_padded_sequence(
            keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_rnn_outputs, _ = self.gru(packed_keys)
        rnn_outputs, _ = pad_packed_sequence(
            packed_rnn_outputs, batch_first=True, padding_value=0, total_length=hist_len
        )

        aux_loss = self.auxiliary_loss(
            rnn_outputs[:, :-1, :], keys[:, 1:, :], neg_keys[:, 1:, :], keys_length - 1
        )

        return rnn_outputs, aux_loss

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, keys_length):
        r"""Computes the auxiliary loss

        Formally:
        ..math: L_{a u x}= \frac{1}{N}(\sum_{i=1}^{N} \sum_{t} \log \sigma(\mathbf{h}_{t}^{i}, \mathbf{e}_{b}^{i}[t+1])
                + \log (1-\sigma(\mathbf{h}_{t}^{i}, \hat{\mathbf{e}}_{b}^{i}[t+1])))

        Args:
            h_states (torch.Tensor): The output of GRUs' hidden layer, [batch_size, history_length - 1, embedding,size].
            click_seq (torch.Tensor): The sequence that users consumed, [batch_size, history_length - 1, embedding,size].
            noclick_seq (torch.Tensor): The sequence that users did not consume, [batch_size, history_length - 1, embedding_size].

         Returns:
            torch.Tensor: auxiliary loss

        """
        batch_size, hist_length, embedding_size = h_states.shape
        click_input = torch.cat([h_states, click_seq], dim=-1)
        noclick_input = torch.cat([h_states, noclick_seq], dim=-1)

        mask = (
            torch.arange(hist_length, device=h_states.device).repeat(batch_size, 1)
            < keys_length.view(-1, 1)
        ).float()
        # click predict
        click_prop = (
            self.auxiliary_net(click_input.view(batch_size * hist_length, -1))
            .view(batch_size, hist_length)[mask > 0]
            .view(-1, 1)
        )
        # click label
        click_target = torch.ones(click_prop.shape, device=click_input.device)

        # non-click predict
        noclick_prop = (
            self.auxiliary_net(noclick_input.view(batch_size * hist_length, -1))
            .view(batch_size, hist_length)[mask > 0]
            .view(-1, 1)
        )
        # non-click label
        noclick_target = torch.zeros(noclick_prop.shape, device=noclick_input.device)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([click_prop, noclick_prop], dim=0),
            torch.cat([click_target, noclick_target], dim=0),
        )

        return loss


class InterestEvolvingLayer(nn.Module):
    """As the joint influence from external environment and internal cognition, different kinds of user interests are
    evolving over time. Interest Evolving Layer can capture interest evolving process that is relative to the target
    item.
    """

    def __init__(
        self,
        mask_mat,
        input_size,
        rnn_hidden_size,
        att_hidden_size=(80, 40),
        activation="sigmoid",
        softmax_stag=True,
        gru="GRU",
    ):
        super(InterestEvolvingLayer, self).__init__()

        self.mask_mat = mask_mat
        self.gru = gru

        if gru == "GRU":
            self.attention_layer = SequenceAttLayer(
                mask_mat, att_hidden_size, activation, softmax_stag, False
            )
            self.dynamic_rnn = nn.GRU(
                input_size=input_size, hidden_size=rnn_hidden_size, batch_first=True
            )

        elif gru == "AIGRU":
            self.attention_layer = SequenceAttLayer(
                mask_mat, att_hidden_size, activation, softmax_stag, True
            )
            self.dynamic_rnn = nn.GRU(
                input_size=input_size, hidden_size=rnn_hidden_size, batch_first=True
            )

        elif gru == "AGRU" or gru == "AUGRU":
            self.attention_layer = SequenceAttLayer(
                mask_mat, att_hidden_size, activation, softmax_stag, True
            )
            self.dynamic_rnn = DynamicRNN(
                input_size=input_size, hidden_size=rnn_hidden_size, gru=gru
            )

    def final_output(self, outputs, keys_length):
        """get the last effective value in the interest evolution sequence
        Args:
            outputs (torch.Tensor): the output of `DynamicRNN` after `pad_packed_sequence`
            keys_length (torch.Tensor): the true length of the user history sequence

        Returns:
            torch.Tensor: The user's CTR for the next item
        """
        batch_size, hist_len, _ = outputs.shape  # [B, T, H]

        mask = torch.arange(hist_len, device=keys_length.device).repeat(
            batch_size, 1
        ) == (keys_length.view(-1, 1) - 1)

        return outputs[mask]

    def forward(self, queries, keys, keys_length):
        hist_len = keys.shape[1]  # T
        keys_length_cpu = keys_length.cpu()
        if self.gru == "GRU":
            packed_keys = pack_padded_sequence(
                input=keys,
                lengths=keys_length_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_rnn_outputs, _ = self.dynamic_rnn(packed_keys)
            rnn_outputs, _ = pad_packed_sequence(
                packed_rnn_outputs,
                batch_first=True,
                padding_value=0.0,
                total_length=hist_len,
            )
            att_outputs = self.attention_layer(queries, rnn_outputs, keys_length)
            outputs = att_outputs.squeeze(1)

        # AIGRU
        elif self.gru == "AIGRU":
            att_outputs = self.attention_layer(queries, keys, keys_length)
            interest = keys * att_outputs.transpose(1, 2)
            packed_rnn_outputs = pack_padded_sequence(
                interest,
                lengths=keys_length_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            _, outputs = self.dynamic_rnn(packed_rnn_outputs)
            outputs = outputs.squeeze(0)

        elif self.gru == "AGRU" or self.gru == "AUGRU":
            att_outputs = self.attention_layer(queries, keys, keys_length).squeeze(
                1
            )  # [B, T]
            packed_rnn_outputs = pack_padded_sequence(
                keys, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False
            )
            packed_att_outputs = pack_padded_sequence(
                att_outputs,
                lengths=keys_length_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            outputs = self.dynamic_rnn(packed_rnn_outputs, packed_att_outputs)
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True, padding_value=0.0, total_length=hist_len
            )
            outputs = self.final_output(outputs, keys_length)  # [B, H]

        return outputs


class AGRUCell(nn.Module):
    """Attention based GRU (AGRU). AGRU uses the attention score to replace the update gate of GRU, and changes the
    hidden state directly.

    Formally:
        ..math: {h}_{t}^{\prime}=\left(1-a_{t}\right) * {h}_{t-1}^{\prime}+a_{t} * \tilde{{h}}_{t}^{\prime}

        :math:`{h}_{t}^{\prime}`, :math:`h_{t-1}^{\prime}`, :math:`{h}_{t-1}^{\prime}`,
        :math: `\tilde{{h}}_{t}^{\prime}` are the hidden state of AGRU

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iu|W_ih)
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        # (W_hr|W_hu|W_hh)
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        if self.bias:
            # (b_ir|b_iu|b_ih)
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            # (b_hr|b_hu|b_hh)
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(self, input, hidden_output, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_output, self.weight_hh, self.bias_hh)
        i_r, i_u, i_h = gi.chunk(3, 1)
        h_r, h_u, h_h = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_u + h_u)
        new_state = torch.tanh(i_h + reset_gate * h_h)

        att_score = att_score.view(-1, 1)
        hy = (1 - att_score) * hidden_output + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU). AUGRU combines attention mechanism and GRU seamlessly.

    Formally:
        ..math: \tilde{{u}}_{t}^{\prime}=a_{t} * {u}_{t}^{\prime} \\
                {h}_{t}^{\prime}=\left(1-\tilde{{u}}_{t}^{\prime}\right) \circ {h}_{t-1}^{\prime}+\tilde{{u}}_{t}^{\prime} \circ \tilde{{h}}_{t}^{\prime}

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iu|W_ih)
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        # (W_hr|W_hu|W_hh)
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        if bias:
            # (b_ir|b_iu|b_ih)
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            # (b_hr|b_hu|b_hh)
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(self, input, hidden_output, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_output, self.weight_hh, self.bias_hh)
        i_r, i_u, i_h = gi.chunk(3, 1)
        h_r, h_u, h_h = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_u + h_u)
        new_state = torch.tanh(i_h + reset_gate * h_h)

        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1 - update_gate) * hidden_output + update_gate * new_state

        return hy


class DynamicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru="AGRU"):
        super(DynamicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru == "AGRU":
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru == "AUGRU":
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, input, att_scores=None, hidden_output=None):
        if not isinstance(input, PackedSequence) or not isinstance(
            att_scores, PackedSequence
        ):
            raise NotImplementedError(
                "DynamicRNN only supports packed input and att_scores"
            )

        input, batch_sizes, sorted_indices, unsorted_indices = input
        att_scores = att_scores.data

        max_batch_size = int(batch_sizes[0])
        if hidden_output is None:
            hidden_output = torch.zeros(
                max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device
            )

        outputs = torch.zeros(
            input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
        )

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                input[begin : begin + batch],
                hidden_output[0:batch],
                att_scores[begin : begin + batch],
            )
            outputs[begin : begin + batch] = new_hx
            hidden_output = new_hx
            begin += batch

        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)
