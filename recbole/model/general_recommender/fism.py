# -*- encoding: utf-8 -*-
# @Time    :   2020/09/28
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com


"""
FISM
#######################################
Reference:
    S. Kabbur et al. "FISM: Factored item similarity models for top-n recommender systems" in KDD 2013

Reference code:
    https://github.com/AaronHeee/Neural-Attentive-Item-Similarity-Model
"""

import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class FISM(GeneralRecommender):
    """FISM is an item-based model for generating top-N recommendations that learns the
    item-item similarity matrix as the product of two low dimensional latent factor matrices.
    These matrices are learned using a structural equation modeling approach, where in the
    value being estimated is not used for its own estimation. 

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(FISM, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']
        # get all users's history interaction information.the history item 
        # matrix is padding by the maximum number of a user's interactions
        self.history_item_matrix, self.history_lens, self.mask_mat = self.get_history_info(dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.reg_weights = config['reg_weights']
        self.alpha = config['alpha']
        self.split_to = config['split_to']

        # split the too large dataset into the specified pieces
        if self.split_to > 0:
            self.group = torch.chunk(torch.arange(self.n_items).to(self.device), self.split_to)

        # define layers and loss
        # construct source and destination item embedding matrix
        self.item_src_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.item_dst_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.user_bias = nn.Parameter(torch.zeros(self.n_users))
        self.item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.bceloss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def get_history_info(self, dataset):
        """get the user history interaction information

        Args:
            dataset (DataSet): train dataset

        Returns:
            tuple: (history_item_matrix, history_lens, mask_mat)

        """
        history_item_matrix, _, history_lens = dataset.history_item_matrix()
        history_item_matrix = history_item_matrix.to(self.device)
        history_lens = history_lens.to(self.device)
        arange_tensor = torch.arange(history_item_matrix.shape[1]).to(self.device)
        mask_mat = (arange_tensor < history_lens.unsqueeze(1)).float()
        return history_item_matrix, history_lens, mask_mat

    def reg_loss(self):
        """calculate the reg loss for embedding layers

        Returns:
            torch.Tensor: reg loss

        """        
        reg_1, reg_2 = self.reg_weights
        loss_1 = reg_1 * self.item_src_embedding.weight.norm(2)
        loss_2 = reg_2 * self.item_dst_embedding.weight.norm(2)

        return loss_1 + loss_2

    def _init_weights(self, module):
        """Initialize the module's parameters

        Note:
            It's a little different from the source code, because pytorch has no function to initialize
            the parameters by truncated normal distribution, so we replace it with xavier normal distribution

        """
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def inter_forward(self, user, item):
        """forward the model by interaction

        """
        user_inter = self.history_item_matrix[user]
        item_num = self.history_lens[user].unsqueeze(1)
        batch_mask_mat = self.mask_mat[user]
        user_history = self.item_src_embedding(user_inter)  # batch_size x max_len x embedding_size
        target = self.item_dst_embedding(item)  # batch_size x embedding_size
        user_bias = self.user_bias[user]  # batch_size x 1
        item_bias = self.item_bias[item]
        similarity = torch.bmm(user_history, target.unsqueeze(2)).squeeze(2)  # batch_size x max_len
        similarity = batch_mask_mat * similarity
        coeff = torch.pow(item_num.squeeze(1), -self.alpha)
        scores = torch.sigmoid(coeff.float() * torch.sum(similarity, dim=1) + user_bias + item_bias)
        return scores

    def user_forward(self, user_input, item_num, user_bias, repeats=None, pred_slc=None):
        """forward the model by user

        Args:
            user_input (torch.Tensor): user input tensor
            item_num (torch.Tensor): user hitory interaction lens
            repeats (int, optional): the number of items to be evaluated
            pred_slc (torch.Tensor, optional): continuous index which controls the current evaluation items,
                                              if pred_slc is None, it will evaluate all items

        Returns:
            torch.Tensor: result

        """
        item_num = item_num.repeat(repeats, 1)
        user_history = self.item_src_embedding(user_input)  # inter_num x embedding_size
        user_history = user_history.repeat(repeats, 1, 1)  # target_items x inter_num x embedding_size
        if pred_slc is None:
            targets = self.item_dst_embedding.weight  # target_items x embedding_size
            item_bias = self.item_bias
        else:
            targets = self.item_dst_embedding(pred_slc)
            item_bias = self.item_bias[pred_slc]
        similarity = torch.bmm(user_history, targets.unsqueeze(2)).squeeze(2)  # inter_num x target_items
        coeff = torch.pow(item_num.squeeze(1), -self.alpha)
        scores = torch.sigmoid(coeff.float() * torch.sum(similarity, dim=1) + user_bias + item_bias)
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

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch_user_bias = self.user_bias[user]
        user_inters = self.history_item_matrix[user]
        item_nums = self.history_lens[user]
        scores = []

        # test users one by one, if the number of items is too large, we will split it to some pieces
        for user_input, item_num, user_bias in zip(user_inters, item_nums.unsqueeze(1), batch_user_bias):
            if self.split_to <= 0:
                output = self.user_forward(user_input[:item_num], item_num, user_bias, repeats=self.n_items)
            else:
                output = []
                for mask in self.group:
                    tmp_output = self.user_forward(user_input[:item_num], item_num, user_bias, repeats=len(mask), pred_slc=mask)
                    output.append(tmp_output)
                output = torch.cat(output, dim=0)
            scores.append(output)
        result = torch.cat(scores, dim=0)
        return result

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward(user, item)
        return output
