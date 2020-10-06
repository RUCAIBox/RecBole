# -*- coding: utf-8 -*-
# @Time   : 2020/9/21
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/2
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
recbox.model.sequential_recommender.caser
################################################

Reference:
Jiaxi Tang et al., "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" in WSDM 2018.

Reference code:
https://github.com/graytowne/caser_pytorch

"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, xavier_normal_, constant_

from recbox.utils import InputType
from recbox.model.loss import RegLoss
from recbox.model.abstract_recommender import SequentialRecommender


class Caser(SequentialRecommender):
    r"""Caser is a model that incorporate CNN for recommendation.
    Note:
        We did not use the sliding window to generate training instances as in the paper, in order that
        the generation method we used is common to other sequential models.
        For comparison with other models, we set the parameter T in the paper as 1.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Caser, self).__init__()

        # load parameters info
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.L = config['L']
        self.embedding_size = config['embedding_size']
        self.n_h = config['nh']
        self.n_v = config['nv']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.item_count = dataset.item_num
        self.user_count = dataset.user_num

        # define activation function and loss
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()
        self.reg_loss = RegLoss()

        # user and item embeddings
        self.user_embedding = nn.Embedding(self.user_count, self.embedding_size)
        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size)

        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, self.embedding_size)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size + self.embedding_size, self.embedding_size)

        # dropout
        self.dropout = nn.Dropout(self.dropout)

        # weight initialization
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 1.0 / module.embedding_dim)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        # Embedding Look-up
        # use unsqueeze() to get a 4-D input for convolution layers. (batchsize * 1 * max_length * embedding_size)
        item_list_emb = self.item_list_embedding(interaction[self.ITEM_ID_LIST]).unsqueeze(1)
        user_emb = self.user_embedding(interaction[self.USER_ID]).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_list_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_list_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        predict_behavior_emb = self.ac_fc(self.fc2(x))

        # the embedding of the predicted item, size:(batch_size * embedding_size)
        return predict_behavior_emb

    def get_item_lookup_table(self):
        r"""Get the transpose of item_list_embedding.weight，size: (embedding_size * item_count)
        Used to calculate the score for each item with the predict_behavior_emb
        """
        return self.item_list_embedding.weight.t()

    def reg_loss_conv_h(self):
        r"""
        L2 loss on conv_h
        """
        loss_conv_h = 0
        for name, parm in self.conv_h.named_parameters():
            if name.endswith('weight'):
                loss_conv_h = loss_conv_h + loss_conv_h * parm.norm(2)
        return self.reg_weight * loss_conv_h

    def calculate_loss(self, interaction):
        target_id = interaction[self.TARGET_ITEM_ID]
        pred = self.forward(interaction)
        logits = torch.matmul(pred, self.get_item_lookup_table())
        loss = self.criterion(logits, target_id)
        reg_loss = self.reg_loss([self.user_embedding.weight, self.item_list_embedding.weight,
                                 self.conv_v.weight,self.fc1.weight, self.fc2.weight])
        loss = loss + self.reg_weight * reg_loss + self.reg_loss_conv_h()
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        pred = self.forward(interaction)
        scores = torch.matmul(pred, self.get_item_lookup_table())
        return scores