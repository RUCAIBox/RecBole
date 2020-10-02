# _*_ coding: utf-8 _*_
# @Time : 2020/9/21
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
Reference:
Feng Xue et al. "Deep Item-based Collaborative Filtering for Top-N Recommendation." in TOIS 2018.
"""

import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbox.utils import InputType
from recbox.model.abstract_recommender import GeneralRecommender
from recbox.model.layers import MLPLayers


class DeepICF(GeneralRecommender):
    """DeepICF is an neural network enhanced item collaborative filter model.
    This implementation is the attention-based version which named DeepICF+a in the original paper.
    For convenience and generalization, any pretrain is not included in this model.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DeepICF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_dim = config['embedding_dim']
        self.attention_dim = config['attention_dim']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.regs = config['regs']
        self.mlp_hidden_size = [self.embedding_dim] + self.mlp_hidden_size
        self.split_to = config['split_to']

        # split the too large dataset into the specified pieces
        if self.split_to > 0:
            self.group = torch.chunk(torch.arange(self.n_items).to(self.device), self.split_to)

        # generate intermediate data

        self.history_item_id, _, self.history_item_len = dataset.history_item_matrix()

        # tensor of shape [n_items, H] where H is max length of history interaction.
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_len = self.history_item_len.to(self.device)
        # define layers

        self.item_embedding_q = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.item_embedding_p = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.att_layers = nn.Linear(self.embedding_dim, self.attention_dim)
        self.h = nn.Linear(self.attention_dim, 1, bias=False)
        self.mlp_layers = MLPLayers(self.mlp_hidden_size, bn=True, init_method='norm')

        self.z = nn.Parameter(torch.randn(self.mlp_hidden_size[-1], 1), requires_grad=True)
        self.bias_u = nn.Embedding(self.n_users, 1)
        self.bias_i = nn.Embedding(self.n_items, 1)

        # define the loss
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def reg_loss(self):
        """Calculate the L2 normalization loss of model parameters.
        Including embedding matrixes and weight matrixes of model.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_1, reg_2 = self.regs[:2]
        loss_1 = reg_1 * self.item_embedding_q.weight.norm(2)
        loss_2 = reg_1 * self.item_embedding_p.weight.norm(2)
        loss_3 = reg_1 * self.att_layers.weight.norm(2)
        for name, parm in self.mlp_layers.named_parameters():
            if name.endswith('weight'):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def forward(self, user, item):
        bi_embedding = self.get_bi_inter_embedding(user, item)
        output = self.mlp_layers(bi_embedding)  # [B, mlp_hidden_size[-1]]

        output = torch.matmul(output, self.z) + self.bias_u(user) + self.bias_i(item)  # [B, 1]
        output = self.sigmoid(output)
        output = output.squeeze()
        return output

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        output = self.forward(user, item)

        loss = self.loss(output, label) + self.reg_loss()
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def get_bi_inter_embedding(self, user, item):
        """Get a batch of pairwise interaction embedding with the user's and item's id .
        Each user has H embedding corresponding to H interacted item (padding when not enough).
        Every embedding interact with target item's embedding in an element wise product and then
        an attention network is applied to get the weighted sum of all H embedding.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id. shape of [B, ]
            item (torch.LongTensor): The input tensor that contains item's id. shape of [B, ]

        Returns:
            bi_inter_embedding(torch.FloatTensor): The embedding tensor of a batch of user. shape of [B, embedding_size]
        """
        item_embedding = self.item_embedding_p(item)  # [B, embedding_size]
        user_embedding = self.item_embedding_q(self.history_item_id[user])  # [B,H,embedding_size]

        item_embedding = item_embedding.unsqueeze(1).repeat_interleave(user_embedding.shape[1], dim=1)  # [B,H,embedding_size]

        pair_wise_inter_embedding = torch.mul(user_embedding, item_embedding)  # [B,H,embedding_size]

        arange_tensor = torch.arange(self.history_item_id.shape[1]).to(self.device)
        mask_mat = (arange_tensor < self.history_item_len[user].unsqueeze(1)).float()  # [B,H]
        att_weight = self.get_att_weight(pair_wise_inter_embedding, mask_mat)  # [B,H]

        # [B,embedding_size]
        bi_inter_embedding = torch.sum(torch.mul(att_weight.unsqueeze(-1), pair_wise_inter_embedding), dim=1)
        coeff = torch.pow(self.history_item_len[user].unsqueeze(1).float()-1, self.alpha)
        bi_inter_embedding = bi_inter_embedding * coeff

        return bi_inter_embedding

    def get_att_weight(self, input, mask=None):
        """"

        """
        input = self.att_layers(input)  # [B,H,attention_dim]
        input = self.h(input)           # [B,H,1]
        input = input.squeeze()         # [B,H]

        exp_input = torch.exp(input)
        if mask is not None:
            exp_input = exp_input * mask                 # [B,H]
        exp_sum = torch.sum(exp_input, dim=1, keepdim=True)  # [B,1]
        exp_sum = torch.pow(exp_sum, self.beta)

        output = torch.div(exp_input, exp_sum)  # [B,H]

        return output

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_inters = self.history_item_id[user]
        item_nums = self.history_item_len[user]
        scores = []

        # test users one by one, if the number of items is too large, we will split it to some pieces
        for user_id, user_input, item_num in zip(user, user_inters, item_nums.unsqueeze(1)):
            if self.split_to <= 0:
                output = self.user_forward(user_id, user_input[:item_num], item_num, repeats=self.n_items)
            else:
                output = []
                for mask in self.group:
                    tmp_output = self.user_forward(user_id, user_input[:item_num], item_num, repeats=len(mask), pred_slc=mask)
                    output.append(tmp_output)
                output = torch.cat(output, dim=0)
            scores.append(output)
        result = torch.cat(scores, dim=0)
        return result

    def user_forward(self, user_id, user_input, inter_num, repeats, pred_slc=None):
        """forward the model by user

        Args:
            user_id (torch.Tensor) : user id tensor
            user_input (torch.Tensor): user input tensor
            inter_num (torch.Tensor): user hitory interaction lens
            repeats (int, optional): the number of items to be evaluated
            pred_slc (torch.Tensor, optional): continuous index which controls the current evaluation items,
                                              if pred_slc is None, it will evaluate all items

        Returns:
            torch.Tensor: result

        """
        user_history = self.item_embedding_q(user_input)  # [inter_num , embedding_size]
        user_history = user_history.unsqueeze(0).repeat_interleave(repeats, dim=0)  # [target_items,inter_num,embedding_size]
        if pred_slc is None:
            targets = self.item_embedding_p.weight  # [target_items , embedding_size]
            item_bias = self.bias_i.weight
        else:
            targets = self.item_embedding_p(pred_slc)
            item_bias = self.bias_i(pred_slc)
        targets = targets.unsqueeze(1).repeat_interleave(inter_num, dim=1)  # [target_items , inter_num, embedding_size]
        pair_wise_inter_embedding = torch.mul(user_history, targets)  # [target_items , inter_num, embedding_size]
        att_weight = self.get_att_weight(pair_wise_inter_embedding)  # [target_items,inter_num]

        # [target_items,embedding_size]
        pair_wise_inter_embedding = torch.sum(torch.mul(att_weight.unsqueeze(-1), pair_wise_inter_embedding), dim=1)
        coeff = torch.pow(self.history_item_len[user_id].float() - 1, self.alpha)
        pair_wise_inter_embedding = pair_wise_inter_embedding * coeff
        output = self.mlp_layers(pair_wise_inter_embedding)  # [target_items, mlp_hidden_size[-1]]

        output = torch.matmul(output, self.z) + self.bias_u(user_id) + item_bias  # [target_items, 1]
        output = self.sigmoid(output)
        output = output.squeeze()
        return output

