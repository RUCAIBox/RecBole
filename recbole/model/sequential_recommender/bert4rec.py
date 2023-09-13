# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 12:08
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

# UPDATE
# @Time   : 2023/9/4
# @Author : Enze Liu
# @Email  : enzeeliu@foxmail.com

r"""
BERT4Rec
################################################

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
    In CIKM 2019.

Reference code:
    The authors' tensorflow implementation https://github.com/FeiSun/BERT4Rec

"""

import random

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class BERT4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(BERT4Rec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.mask_ratio = config["mask_ratio"]

        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )  # add mask_token at the last
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
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ["BPR", "CE"]
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

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

    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(
            item_seq.size(0), dtype=torch.long, device=item_seq.device
        )  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]
        return item_seq

    def forward(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        return output  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        pos_items = interaction[self.POS_ITEMS]
        neg_items = interaction[self.NEG_ITEMS]
        masked_index = interaction[self.MASK_INDEX]

        seq_output = self.forward(masked_item_seq)
        pred_index_map = self.multi_hot_embed(
            masked_index, masked_item_seq.size(-1)
        )  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(
            masked_index.size(0), masked_index.size(1), -1
        )  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        if self.loss_type == "BPR":
            pos_items_emb = self.item_embedding(pos_items)  # [B mask_len H]
            neg_items_emb = self.item_embedding(neg_items)  # [B mask_len H]
            pos_score = (
                torch.sum(seq_output * pos_items_emb, dim=-1)
                + self.output_bias[pos_items]
            )  # [B mask_len]
            neg_score = (
                torch.sum(seq_output * neg_items_emb, dim=-1)
                + self.output_bias[neg_items]
            )  # [B mask_len]
            targets = (masked_index > 0).float()
            loss = -torch.sum(
                torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets
            ) / torch.sum(targets)
            return loss

        elif self.loss_type == "CE":
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            test_item_emb = self.item_embedding.weight[: self.n_items]  # [item_num H]
            logits = (
                torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                + self.output_bias
            )  # [B mask_len item_num]
            targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

            loss = torch.sum(
                loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1))
                * targets
            ) / torch.sum(targets)
            return loss
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_item_emb = self.item_embedding(test_item)
        scores = (torch.mul(seq_output, test_item_emb)).sum(dim=1) + self.output_bias[
            test_item
        ]  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_items_emb = self.item_embedding.weight[
            : self.n_items
        ]  # delete masked token
        scores = (
            torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
        )  # [B, item_num]
        return scores
