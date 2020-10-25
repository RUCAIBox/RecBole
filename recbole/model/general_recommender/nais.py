# -*- encoding: utf-8 -*-
# @Time    :   2020/09/01
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE:
# @Time   : 2020/10/14
# @Author : Kaiyuan Li
# @Email  : tsotfsk@outlook.com

"""
NAIS
######################################
Reference:
    Xiangnan He et al. "NAIS: Neural Attentive Item Similarity Model for Recommendation." in TKDE 2018.

Reference code:
    https://github.com/AaronHeee/Neural-Attentive-Item-Similarity-Model
"""

from logging import getLogger

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType
from torch.nn.init import constant_, normal_, xavier_normal_


class NAIS(GeneralRecommender):
    """NAIS is an attention network, which is capable of distinguishing which historical items
    in a user profile are more important for a prediction. We just implement the model following
    the original author with a pointwise training mode.

    Note:
        instead of forming a minibatch as all training instances of a randomly sampled user which is
        mentioned in the original paper, we still train the model by a randomly sampled interactions.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NAIS, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']
        self.logger = getLogger()

        # get all users's history interaction information.the history item 
        # matrix is padding by the maximum number of a user's interactions
        self.history_item_matrix, self.history_lens, self.mask_mat = self.get_history_info(dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.weight_size = config['weight_size']
        self.algorithm = config['algorithm']
        self.reg_weights = config['reg_weights']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.split_to = config['split_to']
        self.pretrain_path = config['pretrain_path']

        # split the too large dataset into the specified pieces
        if self.split_to > 0:
            self.logger.info('split the n_items to {} pieces'.format(self.split_to))
            self.group = torch.chunk(torch.arange(self.n_items).to(self.device), self.split_to)

        # define layers and loss
        # construct source and destination item embedding matrix
        self.item_src_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.item_dst_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(self.n_items))
        if self.algorithm == 'concat':
            self.mlp_layers = MLPLayers([self.embedding_size*2, self.weight_size])
        elif self.algorithm == 'prod':
            self.mlp_layers = MLPLayers([self.embedding_size, self.weight_size])
        else:
            raise ValueError("NAIS just support attention type in ['concat', 'prod'] but get {}".format(self.algorithm))
        self.weight_layer = nn.Parameter(torch.ones(self.weight_size, 1))
        self.bceloss = nn.BCELoss()

        # parameters initialization
        if self.pretrain_path is not None:
            self.logger.info('use pretrain from [{}]...'.format(self.pretrain_path))
            self._load_pretrain()
        else:
            self.logger.info('unuse pretrain...')
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the module's parameters

        Note:
            It's a little different from the source code, because pytorch has no function to initialize
            the parameters by truncated normal distribution, so we replace it with xavier normal distribution

        """
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def _load_pretrain(self):
        """A simple implementation of loading pretrained parameters.

        """
        fism = torch.load(self.pretrain_path)['state_dict']
        self.item_src_embedding.weight.data.copy_(fism['item_src_embedding.weight'])
        self.item_dst_embedding.weight.data.copy_(fism['item_dst_embedding.weight'])
        for name, parm in self.mlp_layers.named_parameters():
            if name.endswith('weight'):
                xavier_normal_(parm.data)
            elif name.endswith('bias'):
                constant_(parm.data, 0)

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
        """calculate the reg loss for embedding layers and mlp layers

        Returns:
            torch.Tensor: reg loss

        """  
        reg_1, reg_2, reg_3 = self.reg_weights
        loss_1 = reg_1 * self.item_src_embedding.weight.norm(2)
        loss_2 = reg_2 * self.item_dst_embedding.weight.norm(2)
        loss_3 = 0
        for name, parm in self.mlp_layers.named_parameters():
            if name.endswith('weight'):
                loss_3 = loss_3 + reg_3 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def attention_mlp(self, inter, target):
        """layers of attention which support `prod` and `concat`

        Args:
            inter (torch.Tensor): the embedding of history items
            target (torch.Tensor): the embedding of target items

        Returns:
            torch.Tensor: the result of attention

        """
        if self.algorithm == 'prod':
            mlp_input = inter * target.unsqueeze(1)  # batch_size x max_len x embedding_size
        else:
            mlp_input = torch.cat([inter, target.unsqueeze(1).expand_as(inter)], dim=2)  # batch_size x max_len x embedding_size*2
        mlp_output = self.mlp_layers(mlp_input)  # batch_size x max_len x weight_size

        logits = torch.matmul(mlp_output, self.weight_layer).squeeze(2)  # batch_size x max_len
        return logits

    def mask_softmax(self, similarity, logits, bias, item_num, batch_mask_mat):
        """softmax the unmasked user history items and get the final output

        Args:
            similarity (torch.Tensor): the similarity between the histoy items and target items
            logits (torch.Tensor): the initial weights of the history items
            item_num (torch.Tensor): user hitory interaction lengths
            bias (torch.Tensor): bias
            batch_mask_mat (torch.Tensor): the mask of user history interactions

        Returns:
            torch.Tensor: final output

        """
        exp_logits = torch.exp(logits)  # batch_size x max_len

        exp_logits = batch_mask_mat * exp_logits   # batch_size x max_len
        exp_sum = torch.sum(exp_logits, dim=1, keepdim=True)
        exp_sum = torch.pow(exp_sum, self.beta)
        weights = torch.div(exp_logits, exp_sum)

        coeff = torch.pow(item_num.squeeze(1), -self.alpha)
        output = torch.sigmoid(coeff.float() * torch.sum(weights * similarity, dim=1) + bias)

        return output

    def softmax(self, similarity, logits, item_num, bias):
        """softmax the user history features and get the final output

        Args:
            similarity (torch.Tensor): the similarity between the histoy items and target items
            logits (torch.Tensor): the initial weights of the history items
            item_num (torch.Tensor): user hitory interaction lengths
            bias (torch.Tensor): bias

        Returns:
            torch.Tensor: final output

        """
        exp_logits = torch.exp(logits)  # batch_size x max_len
        exp_sum = torch.sum(exp_logits, dim=1, keepdim=True)
        exp_sum = torch.pow(exp_sum, self.beta)
        weights = torch.div(exp_logits, exp_sum)
        coeff = torch.pow(item_num.squeeze(1), -self.alpha)
        output = torch.sigmoid(coeff.float() * torch.sum(weights * similarity, dim=1) + bias)

        return output

    def inter_forward(self, user, item):
        """forward the model by interaction

        """
        user_inter = self.history_item_matrix[user]
        item_num = self.history_lens[user].unsqueeze(1)
        batch_mask_mat = self.mask_mat[user]
        user_history = self.item_src_embedding(user_inter)  # batch_size x max_len x embedding_size
        target = self.item_dst_embedding(item)  # batch_size x embedding_size
        bias = self.bias[item]  # batch_size x 1
        similarity = torch.bmm(user_history, target.unsqueeze(2)).squeeze(2)  # batch_size x max_len
        logits = self.attention_mlp(user_history, target)
        scores = self.mask_softmax(similarity, logits, bias, item_num, batch_mask_mat)
        return scores

    def user_forward(self, user_input, item_num, repeats=None, pred_slc=None):
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

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_inters = self.history_item_matrix[user]
        item_nums = self.history_lens[user]
        scores = []

        # test users one by one, if the number of items is too large, we will split it to some pieces
        for user_input, item_num in zip(user_inters, item_nums.unsqueeze(1)):
            if self.split_to <= 0:
                output = self.user_forward(user_input[:item_num], item_num, repeats=self.n_items)
            else:
                output = []
                for mask in self.group:
                    tmp_output = self.user_forward(user_input[:item_num], item_num, repeats=len(mask), pred_slc=mask)
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
