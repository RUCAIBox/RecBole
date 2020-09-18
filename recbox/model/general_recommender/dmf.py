# _*_ coding: utf-8 _*_
# @Time : 2020/8/21
# @Author : Kaizhou Zhang
# @Email  : kaizhou361@163.com

# UPDATE
# @Time   : 2020/08/31  2020/09/18
# @Author : Kaiyuan Li  Zihan Lin
# @email  : tsotfsk@outlook.com  linzihan.super@foxmail.con

"""
Reference:
Hong-Jian Xue et al. "Deep Matrix Factorization Models for Recommender Systems." in IJCAI 2017.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from recbox.utils import InputType
from recbox.model.abstract_recommender import GeneralRecommender
from recbox.model.layers import MLPLayers


class DMF(GeneralRecommender):
    """DMF is an neural network enhanced matrix factorization model.
    The original interaction matrix of n_users × n_items is set as model input,
    we carefully design the data interface and use sparse tensor to train and test efficiently.
    We just implement the model following the original author with a pointwise training mode.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']

        # load parameters info
        self.user_layers_dim = config['user_layers_dim']
        self.item_layers_dim = config['item_layers_dim']
        # The dimensions of the last layer of users and items must be the same
        assert self.user_layers_dim[-1] == self.item_layers_dim[-1]
        self.min_y_hat = torch.tensor([config['min_y_hat']]).to(self.device)
        self.inter_matrix_type = config['inter_matrix_type']

        # generate intermediate data
        if self.inter_matrix_type == '01':
            self.history_user_id, self.history_user_value, _ = dataset.history_user_matrix()
            self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
            self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        elif self.inter_matrix_type == 'rating':
            self.history_user_id, self.history_user_value, _ = dataset.history_user_matrix(value_field=self.RATING)
            self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix(value_field=self.RATING)
        else:
            raise ValueError("The inter_matrix_type must in ['01', 'rating'] but get {}".format(self.inter_matrix_type))
        self.max_rating = self.history_user_value.max()

        # tensor of shape [n_items, H] where H is max length of history interaction.
        self.history_user_id = self.history_user_id.to(self.device)
        self.history_user_value = self.history_user_value.to(self.device)
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)
        # define layers

        self.user_linear = nn.Linear(in_features=self.n_items, out_features=self.user_layers_dim[0], bias=False)
        self.item_linear = nn.Linear(in_features=self.n_users, out_features=self.item_layers_dim[0], bias=False)
        self.user_fc_layers = MLPLayers(self.user_layers_dim)
        self.item_fc_layers = MLPLayers(self.item_layers_dim)
        # Save the item embedding before dot product layer to speed up evaluation
        self.i_embedding = None

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

    def forward(self, user, item):

        user = self.get_user_embedding(user)

        # Following lines construct tensor of shape [B,n_users] using the tensor of shape [B,H]
        col_indices = self.history_user_id[item].flatten()
        row_indices = torch.arange(item.shape[0]).to(self.device).repeat_interleave(self.history_user_id.shape[1], dim=0)
        matrix_01 = torch.zeros(1).to(self.device).repeat(item.shape[0], self.n_users)
        matrix_01.index_put_((row_indices, col_indices), self.history_user_value[item].flatten())
        item = self.item_linear(matrix_01)

        user = self.user_fc_layers(user)
        item = self.item_fc_layers(item)

        user = F.normalize(user, p=2, dim=1)   # after normalize the vector, cosine distance reduced to dot product.
        item = F.normalize(item, p=2, dim=1)
        vector = torch.mul(user, item).sum(dim=1)

        vector = torch.max(vector, self.min_y_hat)  # restrict the result to [0, 1].
        return vector

    def calculate_loss(self, interaction):
        # when starting a new epoch, the item embedding we saved must be cleared.
        if self.training:
            self.i_embedding = None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.inter_matrix_type == '01':
            label = interaction[self.LABEL]
        elif self.inter_matrix_type == 'rating':
            label = interaction[self.RATING] * interaction[self.LABEL]
        output = self.forward(user, item)

        label = label / self.max_rating  # normalize the label to calculate BCE loss.
        loss = -(label * (output.log()) + (1 - label) * ((1 - output).log())).mean()
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def get_user_embedding(self, user):
        """Get a batch of user's embedding with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id. shape of [B, ]

        Returns:
            output(torch.FloatTensor): The embedding tensor of a batch of user. shape of [B, embedding_size]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).to(self.device).repeat_interleave(self.history_item_id.shape[1],
                                                                                    dim=0)
        matrix_01 = torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        matrix_01.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        user = self.user_linear(matrix_01)

        return user

    def get_item_embedding(self):
        """Get all item's embedding with history interaction matrix.

        Considering the RAM of device, we use matrix multiply on sparse tensor for generalization.

        Returns:
            output(torch.FloatTensor): The embedding tensor of all item. shape of [n_items, embedding_size]
        """
        interaction_matrix = self.interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(interaction_matrix.data)
        item_matrix = torch.sparse.FloatTensor(i, data).to(self.device).transpose(0, 1)
        
        item = torch.sparse.mm(item_matrix, self.item_linear.weight.t())

        item = self.item_fc_layers(item)
        return item

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_embedding = self.get_user_embedding(user)
        u_embedding = self.user_fc_layers(u_embedding)
        u_embedding = F.normalize(u_embedding, p=2, dim=1)

        if self.i_embedding is None:
            self.i_embedding = self.get_item_embedding()
            self.i_embedding = F.normalize(self.i_embedding, p=2, dim=1)

        cos_similarity = torch.mm(u_embedding, self.i_embedding.t())
        cos_similarity = torch.max(cos_similarity, self.min_y_hat)
        return cos_similarity.view(-1)
