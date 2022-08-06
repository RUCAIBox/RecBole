# -*- coding: utf-8 -*-

r"""
CORE
################################################
Reference:
    Yupeng Hou, Binbin Hu, Zhiqiang Zhang, Wayne Xin Zhao. "CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space." in SIGIR 2022.

    https://github.com/RUCAIBox/CORE
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class TransNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["embedding_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]

        self.position_embedding = nn.Embedding(
            dataset.field2seqlen[config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]],
            self.hidden_size,
        )
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CORE(SequentialRecommender):
    r"""CORE is a simple and effective framewor, which unifies the representation spac
    for both the encoding and decoding processes in session-based recommendation.
    """

    def __init__(self, config, dataset):
        super(CORE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.loss_type = config["loss_type"]

        self.dnn_type = config["dnn_type"]
        self.sess_dropout = nn.Dropout(config["sess_dropout"])
        self.item_dropout = nn.Dropout(config["item_dropout"])
        self.temperature = config["temperature"]

        # item embedding
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        # DNN
        if self.dnn_type == "trm":
            self.net = TransNet(config, dataset)
        elif self.dnn_type == "ave":
            self.net = self.ave_net
        else:
            raise ValueError(
                f"dnn_type should be either trm or ave, but have [{self.dnn_type}]."
            )

        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @staticmethod
    def ave_net(item_seq, item_emb):
        mask = item_seq.gt(0)
        alpha = mask.to(torch.float) / mask.sum(dim=-1, keepdim=True)
        return alpha.unsqueeze(-1)

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        # Representation-Consistent Encoder (RCE)
        alpha = self.net(item_seq, x)
        seq_output = torch.sum(alpha * x, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        # Robust Distance Measuring (RDM)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = (
            torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        )
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) / self.temperature
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding.weight
        # no dropout for evaluation
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = (
            torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        )
        return scores
