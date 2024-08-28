# -*- coding: utf-8 -*-
# @Time   : 2022/3/25 13:38
# @Author : HaoJun Qin
# @Email  : 18697951462@163.com

r"""
SimpleX
################################################

Reference:
    Kelong Mao et al. "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." in CIKM 2021.

Reference code:
    https://github.com/xue-pai/TwoToweRS
"""
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class SimpleX(GeneralRecommender):
    r"""SimpleX is a simple, unified collaborative filtering model.

    SimpleX presents a simple and easy-to-understand model. Its advantage lies
    in its loss function, which uses a larger number of negative samples and
    sets a threshold to filter out less informative samples, it also uses
    relative weights to control the balance of positive-sample loss
    and negative-sample loss.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimpleX, self).__init__(config, dataset)

        # Get user history interacted items
        self.history_item_id, _, self.history_item_len = dataset.history_item_matrix(
            max_history_len=config["history_len"]
        )
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_len = self.history_item_len.to(self.device)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.margin = config["margin"]
        self.negative_weight = config["negative_weight"]
        self.gamma = config["gamma"]
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"]
        self.reg_weight = config["reg_weight"]
        self.aggregator = config["aggregator"]
        if self.aggregator not in ["mean", "user_attention", "self_attention"]:
            raise ValueError(
                "aggregator must be mean, user_attention or self_attention"
            )
        self.history_len = torch.max(self.history_item_len, dim=0)

        # user embedding matrix
        self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
        # item embedding matrix
        self.item_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # feature space mapping matrix of user and item
        self.UI_map = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )
            if self.aggregator == "self_attention":
                self.W_q = nn.Linear(self.embedding_size, 1, bias=False)
        # dropout
        self.dropout = nn.Dropout(config["dropout_prob"])
        self.require_pow = config["require_pow"]
        # l2 regularization loss
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        # get the mask
        self.item_emb.weight.data[0, :] = 0

    def get_UI_aggregation(self, user_e, history_item_e, history_len):
        r"""Get the combined vector of user and historically interacted items

        Args:
            user_e (torch.Tensor): User's feature vector, shape: [user_num, embedding_size]
            history_item_e (torch.Tensor): History item's feature vector,
                shape: [user_num, max_history_len, embedding_size]
            history_len (torch.Tensor): User's history length, shape: [user_num]

        Returns:
            torch.Tensor: Combined vector of user and item sequences, shape: [user_num, embedding_size]
        """
        if self.aggregator == "mean":
            pos_item_sum = history_item_e.sum(dim=1)
            # [user_num, embedding_size]
            out = pos_item_sum / (history_len + 1.0e-10).unsqueeze(1)
        elif self.aggregator in ["user_attention", "self_attention"]:
            # [user_num, max_history_len, embedding_size]
            key = self.W_k(history_item_e)
            if self.aggregator == "user_attention":
                # [user_num, max_history_len]
                attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2)
            elif self.aggregator == "self_attention":
                # [user_num, max_history_len]
                attention = self.W_q(key).squeeze(2)
            e_attention = torch.exp(attention)
            mask = (history_item_e.sum(dim=-1) != 0).int()
            e_attention = e_attention * mask
            # [user_num, max_history_len]
            attention_weight = e_attention / (
                e_attention.sum(dim=1, keepdim=True) + 1.0e-10
            )
            # [user_num, embedding_size]
            out = torch.matmul(attention_weight.unsqueeze(1), history_item_e).squeeze(1)
        # Combined vector of user and item sequences
        out = self.UI_map(out)
        g = self.gamma
        UI_aggregation_e = g * user_e + (1 - g) * out
        return UI_aggregation_e

    def get_cos(self, user_e, item_e):
        r"""Get the cosine similarity between user and item

        Args:
            user_e (torch.Tensor): User's feature vector, shape: [user_num, embedding_size]
            item_e (torch.Tensor): Item's feature vector,
                shape: [user_num, item_num, embedding_size]

        Returns:
            torch.Tensor: Cosine similarity between user and item, shape: [user_num, item_num]
        """
        user_e = F.normalize(user_e, dim=1)
        # [user_num, embedding_size, 1]
        user_e = user_e.unsqueeze(2)
        item_e = F.normalize(item_e, dim=2)
        UI_cos = torch.matmul(item_e, user_e)
        return UI_cos.squeeze(2)

    def forward(self, user, pos_item, history_item, history_len, neg_item_seq):
        r"""Get the loss

        Args:
            user (torch.Tensor): User's id, shape: [user_num]
            pos_item (torch.Tensor): Positive item's id, shape: [user_num]
            history_item (torch.Tensor): Id of historty item, shape: [user_num, max_history_len]
            history_len (torch.Tensor): History item's length, shape: [user_num]
            neg_item_seq (torch.Tensor): Negative item seq's id, shape: [user_num, neg_seq_len]

        Returns:
            torch.Tensor: Loss, shape: []
        """
        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        pos_item_e = self.item_emb(pos_item)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)
        # [nuser_num, neg_seq_len, embedding_size]
        neg_item_seq_e = self.item_emb(neg_item_seq)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)
        UI_aggregation_e = self.dropout(UI_aggregation_e)

        pos_cos = self.get_cos(UI_aggregation_e, pos_item_e.unsqueeze(1))
        neg_cos = self.get_cos(UI_aggregation_e, neg_item_seq_e)

        # CCL loss
        pos_loss = torch.relu(1 - pos_cos)
        neg_loss = torch.relu(neg_cos - self.margin)
        neg_loss = neg_loss.mean(1, keepdim=True) * self.negative_weight
        CCL_loss = (pos_loss + neg_loss).mean()

        # l2 regularization loss
        reg_loss = self.reg_loss(
            user_e,
            pos_item_e,
            history_item_e,
            neg_item_seq_e,
            require_pow=self.require_pow,
        )

        loss = CCL_loss + self.reg_weight * reg_loss.sum()
        return loss

    def calculate_loss(self, interaction):
        r"""Data processing and call function forward(), return loss

        To use SimpleX, a user must have a historical transaction record,
        a pos item and a sequence of neg items. Based on the RecBole
        framework, the data in the interaction object is ordered, so
        we can get the data quickly.
        """
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # get the sequence of neg items
        neg_item_seq = neg_item.reshape((self.neg_seq_len, -1))
        neg_item_seq = neg_item_seq.T
        user_number = int(len(user) / self.neg_seq_len)
        # user's id
        user = user[0:user_number]
        # historical transaction record
        history_item = self.history_item_id[user]
        # positive item's id
        pos_item = pos_item[0:user_number]
        # history_len
        history_len = self.history_item_len[user]

        loss = self.forward(user, pos_item, history_item, history_len, neg_item_seq)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        history_item = self.history_item_id[user]
        history_len = self.history_item_len[user]
        test_item = interaction[self.ITEM_ID]

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        test_item_e = self.item_emb(test_item)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        UI_cos = self.get_cos(UI_aggregation_e, test_item_e.unsqueeze(1))
        return UI_cos.squeeze(1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        history_item = self.history_item_id[user]
        history_len = self.history_item_len[user]

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        UI_aggregation_e = F.normalize(UI_aggregation_e, dim=1)
        all_item_emb = self.item_emb.weight
        all_item_emb = F.normalize(all_item_emb, dim=1)
        UI_cos = torch.matmul(UI_aggregation_e, all_item_emb.T)
        return UI_cos
