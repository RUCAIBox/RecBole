# -*- coding: utf-8 -*-
# @Time     : 2020/11/22 12:08
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun
# @Email    : shaoweiqi@ruc.edu.cn

r"""
HRM
################################################

Reference:
    Pengfei Wang et al. "Learning Hierarchical Representation Model for Next Basket Recommendation." in SIGIR 2015.

Reference code:
    https://github.com/wubinzzu/NeuRec

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class HRM(SequentialRecommender):
    r"""
    HRM can well capture both sequential behavior and users’ general taste by involving transaction and
    user representations in prediction.

    HRM user max- & average- pooling as a good helper.
    """

    def __init__(self, config, dataset):
        super(HRM, self).__init__(config, dataset)

        # load the dataset information
        self.n_user = dataset.num(self.USER_ID)
        self.device = config["device"]

        # load the parameters information
        self.embedding_size = config["embedding_size"]
        self.pooling_type_layer_1 = config["pooling_type_layer_1"]
        self.pooling_type_layer_2 = config["pooling_type_layer_2"]
        self.high_order = config["high_order"]
        assert (
            self.high_order <= self.max_seq_length
        ), "high_order can't longer than the max_seq_length"
        self.reg_weight = config["reg_weight"]
        self.dropout_prob = config["dropout_prob"]

        # define the layers and loss type
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.user_embedding = nn.Embedding(self.n_user, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.loss_type = config["loss_type"]
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameters of the model
        self.apply(self._init_weights)

    def inverse_seq_item(self, seq_item, seq_item_len):
        """
        inverse the seq_item, like this
            [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]
        """
        seq_item = seq_item.cpu().numpy()
        seq_item_len = seq_item_len.cpu().numpy()
        new_seq_item = []
        for items, length in zip(seq_item, seq_item_len):
            item = list(items[:length])
            zeros = list(items[length:])
            seqs = zeros + item
            new_seq_item.append(seqs)
        seq_item = torch.tensor(new_seq_item, dtype=torch.long, device=self.device)

        return seq_item

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, seq_item, user, seq_item_len):
        # seq_item=self.inverse_seq_item(seq_item)
        seq_item = self.inverse_seq_item(seq_item, seq_item_len)

        seq_item_embedding = self.item_embedding(seq_item)
        # batch_size * seq_len * embedding_size

        high_order_item_embedding = seq_item_embedding[:, -self.high_order :, :]
        # batch_size * high_order * embedding_size

        user_embedding = self.dropout(self.user_embedding(user))
        # batch_size * embedding_size

        # layer 1
        if self.pooling_type_layer_1 == "max":
            high_order_item_embedding = torch.max(
                high_order_item_embedding, dim=1
            ).values
            # batch_size * embedding_size
        else:
            for idx, len in enumerate(seq_item_len):
                if len > self.high_order:
                    seq_item_len[idx] = self.high_order
            high_order_item_embedding = torch.sum(seq_item_embedding, dim=1)
            high_order_item_embedding = torch.div(
                high_order_item_embedding, seq_item_len.unsqueeze(1).float()
            )
            # batch_size * embedding_size
        hybrid_user_embedding = self.dropout(
            torch.cat(
                [
                    user_embedding.unsqueeze(dim=1),
                    high_order_item_embedding.unsqueeze(dim=1),
                ],
                dim=1,
            )
        )
        # batch_size * 2_mul_embedding_size

        # layer 2
        if self.pooling_type_layer_2 == "max":
            hybrid_user_embedding = torch.max(hybrid_user_embedding, dim=1).values
            # batch_size * embedding_size
        else:
            hybrid_user_embedding = torch.mean(hybrid_user_embedding, dim=1)
            # batch_size * embedding_size

        return hybrid_user_embedding

    def calculate_loss(self, interaction):
        seq_item = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(seq_item, user, seq_item_len)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_emb = self.item_embedding(pos_items)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight.t()
            logits = torch.matmul(seq_output, test_item_emb)
            loss = self.loss_fct(logits, pos_items)

            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user, seq_item_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user, seq_item_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))

        return scores
