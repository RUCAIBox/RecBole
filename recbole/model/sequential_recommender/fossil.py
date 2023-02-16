# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 20:00
# @Author  : Shao Weiqi
# @Reviewer : Lin Kun
# @Email   : shaoweiqi@ruc.edu.cn

r"""
FOSSIL
################################################

Reference:
    Ruining He et al. "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation." in ICDM 2016.


"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class FOSSIL(SequentialRecommender):
    r"""
    FOSSIL uses similarity of the items as main purpose and uses high MC as a way of sequential preference improve of
    ability of sequential recommendation

    """

    def __init__(self, config, dataset):
        super(FOSSIL, self).__init__(config, dataset)

        # load the dataset information
        self.n_users = dataset.num(self.USER_ID)
        self.device = config["device"]

        # load the parameters
        self.embedding_size = config["embedding_size"]
        self.order_len = config["order_len"]
        assert (
            self.order_len <= self.max_seq_length
        ), "order_len can't longer than the max_seq_length"
        self.reg_weight = config["reg_weight"]
        self.alpha = config["alpha"]

        # define the layers and loss type
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.user_lambda = nn.Embedding(self.n_users, self.order_len)
        self.lambda_ = nn.Parameter(torch.zeros(self.order_len))

        self.loss_type = config["loss_type"]
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameters of the model
        self.apply(self.init_weights)

    def inverse_seq_item_embedding(self, seq_item_embedding, seq_item_len):
        """
        inverse seq_item_embedding like this (simple to 2-dim):

        [1,2,3,0,0,0] -- ??? -- >> [0,0,0,1,2,3]

        first: [0,0,0,0,0,0] concat [1,2,3,0,0,0]

        using gather_indexes: to get one by one

        first get 3,then 2,last 1
        """
        zeros = torch.zeros_like(seq_item_embedding, dtype=torch.float).to(self.device)
        # batch_size * seq_len * embedding_size
        item_embedding_zeros = torch.cat([zeros, seq_item_embedding], dim=1)
        # batch_size * 2_mul_seq_len * embedding_size
        embedding_list = list()
        for i in range(self.order_len):
            embedding = self.gather_indexes(
                item_embedding_zeros,
                self.max_seq_length + seq_item_len - self.order_len + i,
            )
            embedding_list.append(embedding.unsqueeze(1))
        short_item_embedding = torch.cat(embedding_list, dim=1)
        # batch_size * short_len * embedding_size

        return short_item_embedding

    def reg_loss(self, user_embedding, item_embedding, seq_output):
        reg_1 = self.reg_weight
        loss_1 = (
            reg_1 * torch.norm(user_embedding, p=2)
            + reg_1 * torch.norm(item_embedding, p=2)
            + reg_1 * torch.norm(seq_output, p=2)
        )

        return loss_1

    def init_weights(self, module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)

    def forward(self, seq_item, seq_item_len, user):
        seq_item_embedding = self.item_embedding(seq_item)

        high_order_seq_item_embedding = self.inverse_seq_item_embedding(
            seq_item_embedding, seq_item_len
        )
        # batch_size * order_len * embedding

        high_order = self.get_high_order_Markov(high_order_seq_item_embedding, user)
        similarity = self.get_similarity(seq_item_embedding, seq_item_len)

        return high_order + similarity

    def get_high_order_Markov(self, high_order_item_embedding, user):
        """

        in order to get the inference of past items and the user's taste to the current predict item
        """

        user_lambda = self.user_lambda(user).unsqueeze(dim=2)
        # batch_size * order_len * 1
        lambda_ = self.lambda_.unsqueeze(dim=0).unsqueeze(dim=2)
        # 1 * order_len * 1
        lambda_ = torch.add(user_lambda, lambda_)
        # batch_size * order_len * 1
        high_order_item_embedding = torch.mul(high_order_item_embedding, lambda_)
        # batch_size * order_len * embedding_size
        high_order_item_embedding = high_order_item_embedding.sum(dim=1)
        # batch_size * embedding_size

        return high_order_item_embedding

    def get_similarity(self, seq_item_embedding, seq_item_len):
        """
        in order to get the inference of past items to the current predict item
        """
        coeff = torch.pow(seq_item_len.unsqueeze(1), -self.alpha).float()
        # batch_size * 1
        similarity = torch.mul(coeff, seq_item_embedding.sum(dim=1))
        # batch_size * embedding_size

        return similarity

    def calculate_loss(self, interaction):
        seq_item = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(seq_item, seq_item_len, user)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_emb = self.item_embedding(pos_items)

        user_lambda = self.user_lambda(user)
        pos_items_embedding = self.item_embedding(pos_items)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss + self.reg_loss(user_lambda, pos_items_embedding, seq_output)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss + self.reg_loss(user_lambda, pos_items_embedding, seq_output)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, item_seq_len, user)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, user)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
