# -*- encoding: utf-8 -*-
# @Time    :   2020/09/01
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

"""
Reference:
Xiangnan He et al., "NAIS: Neural Attentive Item Similarity Model for Recommendation." in TKDE 2018.
Also, our code is base on https://github.com/AaronHeee/Neural-Attentive-Item-Similarity-Model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, xavier_normal_
from torch.nn.utils.rnn import pad_sequence

from ...utils import InputType
from ..abstract_recommender import GeneralRecommender
from ..layers import MLPLayers


class NAIS(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NAIS, self).__init__()
        self.device = config['device']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.LABEL = config['LABEL_FIELD']

        self.pretrain = config['pretrain']
        self.embedding_size = config['embedding_size']
        self.weight_size = config['weight_size']
        self.algorithm = config['algorithm']
        self.regs = config['regs']
        self.alpha = config['alpha']
        self.beta = config['beta']

        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        # Construct src item embedding matrix, padding at n_items position
        self.item_src_embedding = nn.Embedding(self.n_items+1, self.embedding_size, padding_idx=self.n_items)
        # Construct dst item embedding matrix, the target items don't require padding
        self.item_dst_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.bias = nn.Parameter(torch.zeros(self.n_items))

        if self.algorithm == 'concat':
            self.mlp_layers = MLPLayers([self.embedding_size*2, self.weight_size])
        elif self.algorithm == 'prod':
            self.mlp_layers = MLPLayers([self.embedding_size, self.weight_size])
        else:
            raise ValueError("NAIS just support attention type in ['concat', 'prod'] but get {}".format(self.algorithm))
        self.weight_layer = nn.Parameter(torch.ones(self.weight_size, 1))

        self.bceloss = nn.BCELoss()

        self.apply(self.init_weights)

    def reg_loss(self):
        reg_1, reg_2, reg_3 = self.regs
        loss_1 = reg_1 * self.item_src_embedding.weight.norm(2)
        loss_2 = reg_2 * self.item_dst_embedding.weight.norm(2)
        loss_3 = 0
        for parm in self.mlp_layers.parameters():
            loss_3 = loss_3 + reg_3 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def attention_mlp(self, inter, target):
        batch_size, max_num, _ = inter.size()
        if self.algorithm == 'prod':
            mlp_input = inter * target.unsqueeze(1)
        else:
            mlp_input = torch.cat([inter, target.unsqueeze(1).expand_as(inter)], dim=2)
        mlp_output = self.mlp_layers(mlp_input)
        logits = torch.matmul(mlp_output, self.weight_layer).view(batch_size, max_num)
        return logits

    def mask_softmax(self, similarity, logits, bias, item_num):

        # softmax for not mask features
        _, max_len = logits.size()
        exp_logits = torch.exp(logits)
        mask_mat = torch.arange(max_len).to(self.device).view(1, -1).repeat(logits.size(0), 1)
        mask_mat = mask_mat < item_num
        exp_logits = mask_mat.float() * exp_logits
        exp_sum = torch.sum(exp_logits, dim=1, keepdim=True)  # (b, 1)
        exp_sum = torch.pow(exp_sum, self.beta)

        weights = torch.div(exp_logits, exp_sum)
        coeff = torch.pow(item_num.squeeze(1), self.alpha)
        output = torch.sigmoid(coeff * torch.sum(weights * similarity, dim=1) + bias)

        return output

    def init_weights(self, module):
        # It's a little different from the source code, because pytorch has no function to initialize
        # the parameters by truncated normal distribution, so we replace it with xavier normal distribution
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, user, item):
        user_bool_inter = torch.from_numpy(self.interaction_matrix[user.cpu()].toarray()).to(self.device)
        item_num = torch.sum(user_bool_inter, axis=1, keepdim=True)
        _, indices = torch.nonzero(user_bool_inter, as_tuple=True)
        user_inter = torch.split(indices, item_num.view(-1).int().cpu().numpy().tolist(), dim=0)
        user_inter = pad_sequence(user_inter, batch_first=True, padding_value=self.n_items)
        user_history = self.item_src_embedding(user_inter)
        target = self.item_dst_embedding(item)
        bias = self.bias[item]

        similarity = torch.bmm(user_history, target.unsqueeze(2)).squeeze(2)
        logits = self.attention_mlp(user_history, target)
        scores = self.mask_softmax(similarity, logits, bias, item_num)

        return scores

    def calculate_loss(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        output = self.forward(user, item)
        loss = self.bceloss(output, label) + self.reg_loss()

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)
