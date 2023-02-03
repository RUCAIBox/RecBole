# -*- coding: utf-8 -*-
# @Time     : 2020/11/21 16:36
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun
# @Email    : shaoweiqi@ruc.edu.cn

r"""
HGN
################################################

Reference:
    Chen Ma et al. "Hierarchical Gating Networks for Sequential Recommendation."in SIGKDD 2019


"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class HGN(SequentialRecommender):
    r"""
    HGN sets feature gating and instance gating to get the important feature and item for predicting the next item

    """

    def __init__(self, config, dataset):
        super(HGN, self).__init__(config, dataset)

        # load the dataset information
        self.n_user = dataset.num(self.USER_ID)
        self.device = config["device"]

        # load the parameter information
        self.embedding_size = config["embedding_size"]
        self.reg_weight = config["reg_weight"]
        self.pool_type = config["pooling_type"]

        if self.pool_type not in ["max", "average"]:
            raise NotImplementedError("Make sure 'loss_type' in ['max', 'average']!")

        # define the layers and loss function
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.user_embedding = nn.Embedding(self.n_user, self.embedding_size)

        # define the module feature gating need
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.b = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=True)

        # define the module instance gating need
        self.w3 = nn.Linear(self.embedding_size, 1, bias=False)
        self.w4 = nn.Linear(self.embedding_size, self.max_seq_length, bias=False)

        # define item_embedding for prediction
        self.item_embedding_for_prediction = nn.Embedding(
            self.n_items, self.embedding_size
        )

        self.sigmoid = nn.Sigmoid()

        self.loss_type = config["loss_type"]
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameters of the model
        self.apply(self._init_weights)

    def reg_loss(self, user_embedding, item_embedding, seq_item_embedding):
        reg_1, reg_2 = self.reg_weight
        loss_1_part_1 = reg_1 * torch.norm(self.w1.weight, p=2)
        loss_1_part_2 = reg_1 * torch.norm(self.w2.weight, p=2)
        loss_1_part_3 = reg_1 * torch.norm(self.w3.weight, p=2)
        loss_1_part_4 = reg_1 * torch.norm(self.w4.weight, p=2)
        loss_1 = loss_1_part_1 + loss_1_part_2 + loss_1_part_3 + loss_1_part_4

        loss_2_part_1 = reg_2 * torch.norm(user_embedding, p=2)
        loss_2_part_2 = reg_2 * torch.norm(item_embedding, p=2)
        loss_2_part_3 = reg_2 * torch.norm(seq_item_embedding, p=2)
        loss_2 = loss_2_part_1 + loss_2_part_2 + loss_2_part_3

        return loss_1 + loss_2

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0.0, 1 / self.embedding_size)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def feature_gating(self, seq_item_embedding, user_embedding):
        """

        choose the features that will be sent to the next stage(more important feature, more focus)
        """

        batch_size, seq_len, embedding_size = seq_item_embedding.size()
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.w1(seq_item_embedding)
        # batch_size * seq_len * embedding_size
        user_embedding = self.w2(user_embedding)
        # batch_size * embedding_size
        user_embedding = user_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        # batch_size * seq_len * embedding_size

        user_item = self.sigmoid(seq_item_embedding + user_embedding + self.b)
        # batch_size * seq_len * embedding_size

        user_item = torch.mul(seq_item_embedding_value, user_item)
        # batch_size * seq_len * embedding_size

        return user_item

    def instance_gating(self, user_item, user_embedding):
        """

        choose the last click items that will influence the prediction( more important more chance to get attention)
        """

        user_embedding_value = user_item

        user_item = self.w3(user_item)
        # batch_size * seq_len * 1

        user_embedding = self.w4(user_embedding).unsqueeze(2)
        # batch_size * seq_len * 1

        instance_score = self.sigmoid(user_item + user_embedding).squeeze(-1)
        # batch_size * seq_len * 1
        output = torch.mul(instance_score.unsqueeze(2), user_embedding_value)
        # batch_size * seq_len * embedding_size

        if self.pool_type == "average":
            output = torch.div(
                output.sum(dim=1), instance_score.sum(dim=1).unsqueeze(1)
            )
            # batch_size * embedding_size
        else:
            # for max_pooling
            index = torch.max(instance_score, dim=1)[1]
            # batch_size * 1
            output = self.gather_indexes(output, index)
            # batch_size * seq_len * embedding_size ==>> batch_size * embedding_size

        return output

    def forward(self, seq_item, user):
        seq_item_embedding = self.item_embedding(seq_item)
        user_embedding = self.user_embedding(user)
        feature_gating = self.feature_gating(seq_item_embedding, user_embedding)
        instance_gating = self.instance_gating(feature_gating, user_embedding)
        # batch_size * embedding_size
        item_item = torch.sum(seq_item_embedding, dim=1)
        # batch_size * embedding_size

        return user_embedding + instance_gating + item_item

    def calculate_loss(self, interaction):
        seq_item = interaction[self.ITEM_SEQ]
        seq_item_embedding = self.item_embedding(seq_item)
        user = interaction[self.USER_ID]
        user_embedding = self.user_embedding(user)
        seq_output = self.forward(seq_item, user)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_emb = self.item_embedding_for_prediction(pos_items)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss + self.reg_loss(
                user_embedding, pos_items_emb, seq_item_embedding
            )
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding_for_prediction.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss + self.reg_loss(
                user_embedding, pos_items_emb, seq_item_embedding
            )

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user)
        test_item_emb = self.item_embedding_for_prediction(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user)
        test_items_emb = self.item_embedding_for_prediction.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
