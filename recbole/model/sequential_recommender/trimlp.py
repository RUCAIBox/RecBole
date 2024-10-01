# -*- coding: utf-8 -*-
# @Time   : 2024/09/26 12:19
# @Author : Andreas Peintner
# @Email  : anpeintner@gmail.com

r"""
TriMLP
################################################

Reference:
    Jiang et al. "TriMLP: A Foundational MLP-like Architecture for Sequential Recommendation" in TOIS 2024.

Reference code:
    https://github.com/jiangyiheng1/TriMLP/
"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

def global_kernel(seq_len):
    mask = torch.triu(torch.ones([seq_len, seq_len]))
    matrix = torch.ones([seq_len, seq_len])
    matrix = matrix.masked_fill(mask == 0.0, -1e9)
    kernel = nn.parameter.Parameter(matrix, requires_grad=True)
    return kernel


def local_kernel(seq_len, n_session):
    mask = torch.zeros([seq_len, seq_len])
    for i in range(0, seq_len, seq_len // n_session):
        mask[i:i + seq_len // n_session, i:i + seq_len // n_session] = torch.ones(
            [seq_len // n_session, seq_len // n_session])
    mask = torch.triu(mask)
    matrix = torch.ones([seq_len, seq_len])
    matrix = matrix.masked_fill(mask == 0.0, -1e9)
    kernel = nn.parameter.Parameter(matrix, requires_grad=True)
    return kernel

class TriMixer(nn.Module):
    def __init__(self, seq_len, n_session, act=nn.Sigmoid()):
        super().__init__()
        assert seq_len % n_session == 0
        self.l = seq_len
        self.n_s = n_session
        self.act = act
        self.local_mixing = local_kernel(self.l, self.n_s)
        self.global_mixing = global_kernel(self.l)

    def forward(self, x):
        x = torch.matmul(x.permute(0, 2, 1), self.global_mixing.softmax(dim=-1))
        if self.act:
            x = self.act(x)

        x = torch.matmul(x, self.local_mixing.softmax(dim=-1)).permute(0, 2, 1)
        if self.act:
            x = self.act(x)

        return x
    
    def extra_repr(self):
        return f"seq_len={self.l}, n_session={self.n_s}, act={self.act}"


class TriMLP(SequentialRecommender):
    r"""TriMLP: A Foundational MLP-like Architecture for Sequential Recommendation
    """

    def __init__(self, config, dataset):
        super(TriMLP, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.loss_type = config["loss_type"]

        if config["act_fct"] == "sigmoid":
            self.act_fct = nn.Sigmoid()
        elif config["act_fct"] == "tanh":
            self.act_fct = nn.Tanh()
        else:
            self.act_fct = None

        self.dropout_prob = config["dropout_prob"]

        self.num_session = config["num_session"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.mixer = TriMixer(self.max_seq_length, self.num_session, act=self.act_fct)
        self.final_layer = nn.Linear(self.embedding_size, self.embedding_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        mixer_output = self.mixer(item_seq_emb_dropout)
        seq_output = self.gather_indexes(mixer_output, item_seq_len - 1)
        seq_output = self.final_layer(seq_output)

        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
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
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
