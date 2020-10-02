# -*- coding: utf-8 -*-
# @Time   : 2020/9/8 19:24
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/10/2
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

r"""
recbox.model.sequential_recommender.stamp
################################################

Reference:
Qiao Liu et al. "STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation." in KDD 2018.

"""

import torch
from torch import nn
from torch.nn.init import normal_
from recbox.utils import InputType
from recbox.model.abstract_recommender import SequentialRecommender


class STAMP(SequentialRecommender):
    r"""STAMP is capable of capturing users’ general interests from the long-term memory of a session context,
    whilst taking into account users’ current interests from the short-term memory of the last-clicks.


    Note:

            According to the test results, we made a little modification to the score function mentioned in the paper,
            and did not use the final sigmoid activation function.

    """
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(STAMP, self).__init__()
        # load parameters info
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']


        self.embedding_size = config['embedding_size']
        self.item_count = dataset.item_num
        # item embeddings
        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)
        # define weights and bias
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w0 = nn.Linear(self.embedding_size, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=True)
        # define layers,activation and loss
        self.mlp_a = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.mlp_b = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.criterion = nn.CrossEntropyLoss()
        # weight initialization
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def get_item_lookup_table(self):
        r"""Get the transpose of item_list_embedding.weight，Shape of (embedding_size, item_count+padding_id)
        Used to calculate the score for each item with the predict_behavior_emb
        """
        return self.item_list_embedding.weight.t()

    def forward(self, interaction):
        item_list_emb = self.item_list_embedding(interaction[self.ITEM_ID_LIST])
        last_inputs = self.gather_indexes(item_list_emb, interaction[self.ITEM_LIST_LEN] - 1)
        org_memory = item_list_emb
        ms = torch.div(torch.sum(org_memory, dim=1), interaction[self.ITEM_LIST_LEN].unsqueeze(1).float())
        alpha = self.count_alpha(org_memory, last_inputs, ms)
        vec = torch.matmul(alpha.unsqueeze(1), org_memory)
        ma = vec.squeeze(1) + ms
        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last_inputs))
        predict_behavior_emb = hs * ht
        return predict_behavior_emb

    def count_alpha(self, context, aspect, output):
        r"""This is a function that count the attention weights

        Args:
            context(torch.FloatTensor): Item list embedding matrix, shape of [batch_size, time_steps, emb]
            aspect(torch.FloatTensor): The embedding matrix of the last click item, shape of [batch_size, emb]
            output(torch.FloatTensor): The average of the context, shape of [batch_size, emb]

        Returns:
            alpha(torch.FloatTensor):attention weights, shape of [batch_size, time_steps]
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        output_3dim = output.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha

    def calculate_loss(self, interaction):
        target_id = interaction[self.TARGET_ITEM_ID]
        pred = self.forward(interaction)
        logits = torch.matmul(pred, self.get_item_lookup_table())
        loss = self.criterion(logits, target_id)
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        pred = self.forward(interaction)
        scores = torch.matmul(pred, self.get_item_lookup_table())
        return scores
