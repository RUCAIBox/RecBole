# -*- encoding: utf-8 -*-
# @Time    :   2020/09/01
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

"""
Reference:
Xiangnan He et al., "NAIS: Neural Attentive Item Similarity Model for Recommendation." in TKDE 2018.
Also, our code is based on https://github.com/AaronHeee/Neural-Attentive-Item-Similarity-Model
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

        self.embedding_size = config['embedding_size']
        self.weight_size = config['weight_size']
        self.algorithm = config['algorithm']
        self.regs = config['regs']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.split_to = config['split_to']

        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        self.n_items = dataset.item_num

        if self.split_to > 0:
            self.group = torch.chunk(torch.arange(self.n_items).to(self.device), self.split_to)

        # Construct src item embedding matrix, padding at n_items position
        self.item_src_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=self.n_items)
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
        if self.algorithm == 'prod':
            mlp_input = inter * target.unsqueeze(1)  # batch_size x max_len x embedding_size
        else:
            mlp_input = torch.cat([inter, target.unsqueeze(1).expand_as(inter)], dim=2)  # batch_size x max_len x embedding_size*2
        mlp_output = self.mlp_layers(mlp_input)  # batch_size x max_len x weight_size

        logits = torch.matmul(mlp_output, self.weight_layer).squeeze(2)  # batch_size x max_len
        return logits

    def mask_softmax(self, similarity, logits, bias, item_num):

        # softmax for not mask features
        batch_size, max_len = logits.size()
        exp_logits = torch.exp(logits)  # batch_size x max_len

        mask_mat = torch.arange(max_len).to(self.device).view(1, -1).repeat(batch_size, 1)  # batch_size x max_len
        mask_mat = mask_mat < item_num  # batch_size x max_len
        exp_logits = mask_mat.float() * exp_logits   # batch_size x max_len
        exp_sum = torch.sum(exp_logits, dim=1, keepdim=True)
        exp_sum = torch.pow(exp_sum, self.beta)
        weights = torch.div(exp_logits, exp_sum)

        coeff = torch.pow(item_num.squeeze(1), -self.alpha)
        output = torch.sigmoid(coeff * torch.sum(weights * similarity, dim=1) + bias)

        return output

    def softmax(self, similarity, logits, item_num, bias):
        exp_logits = torch.exp(logits)  # batch_size x max_len
        exp_sum = torch.sum(exp_logits, dim=1, keepdim=True)
        exp_sum = torch.pow(exp_sum, self.beta)
        weights = torch.div(exp_logits, exp_sum)
        coeff = torch.pow(item_num.squeeze(1), -self.alpha)
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

    def inter_forward(self, user, item):
        user_inter, item_num = self.get_input(user)
        user_inter = pad_sequence(user_inter, batch_first=True, padding_value=self.n_items)
        user_history = self.item_src_embedding(user_inter)  # batch_size x max_len x embedding_size
        target = self.item_dst_embedding(item)  # batch_size x embedding_size
        bias = self.bias[item]  # batch_size x 1

        similarity = torch.bmm(user_history, target.unsqueeze(2)).squeeze(2)  # batch_size x max_len
        logits = self.attention_mlp(user_history, target)
        scores = self.mask_softmax(similarity, logits, bias, item_num)
        return scores

    def user_forward(self, user_input, item_num, repeats=None, pred_slc=None):
        item_num = item_num.repeat(repeats, 1)
        user_history = self.item_src_embedding(user_input)  # inter_num x embedding_size
        user_history = user_history.repeat(repeats, 1, 1)  # target_items x inter_num x embedding_size
        if pred_slc is None:
            targets = self.item_dst_embedding.weight  # target_items x embedding_size
            bias = self.bias
        else:
            targets = self.item_dst_embedding(pred_slc)
            bias = self.bias[pred_slc]
        similarity = torch.bmm(user_history, targets.unsqueeze(2)).squeeze(2)  # inter_num x target_items
        logits = self.attention_mlp(user_history, targets)
        scores = self.softmax(similarity, logits, item_num, bias)
        return scores

    def forward(self, user, item):
        return self.inter_forward(user, item)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        output = self.forward(user, item)
        loss = self.bceloss(output, label) + self.reg_loss()
        return loss

    def get_input(self, user):
        user_inputs = self.interaction_matrix[user.cpu()]  # batch_size x n_items
        indices = torch.from_numpy(user_inputs.indices).long().to(self.device)
        index_list = user_inputs.indptr.astype(int).tolist()
        slcs = []
        for start, end in zip(index_list[:-1], index_list[1:]):
            slcs.append(end - start)
        user_iters = torch.split(indices, slcs)
        item_nums = torch.FloatTensor(list(map(len, user_iters))).view(-1, 1).to(self.device)  # batch_size x 1
        return user_iters, item_nums

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        user_ids = user.unique()
        user_iters, item_nums = self.get_input(user_ids)
        scores = []
        for user_input, item_num in zip(user_iters, item_nums.squeeze(1)):
            if self.split_to <= 0:
                output = self.user_forward(user_input, item_num, repeats=self.n_items)
            else:
                output = []
                for mask in self.group:
                    tmp_output = self.user_forward(user_input, item_num, repeats=len(mask), pred_slc=mask)
                    output.append(tmp_output)
                output = torch.cat(output, dim=0)
            scores.append(output)
        result = torch.cat(scores, dim=0)
        return result
