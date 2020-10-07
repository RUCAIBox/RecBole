# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE:
# @Time   : 2020/8/19, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

r"""
recbox.model.sequential_recommender.gru4rec
################################################

Reference:
Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""


import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from recbox.utils import InputType
from recbox.model.abstract_recommender import SequentialRecommender


class GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

            Regarding the innovation of this article,we can only achieve the data augmentation mentioned in the paper and directly output the embedding of the item,
            in order that the generation method we used is common to other sequential models.
    """
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(GRU4Rec, self).__init__()
        # load parameters info
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']


        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.item_count = dataset.item_num
        # define layers and loss
        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        self.criterion = nn.CrossEntropyLoss()
        # parameters initialization
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module,nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)

    def get_item_lookup_table(self):
        r"""Get the transpose of item_list_embedding.weight，Shape of (embedding_size, item_count+padding_id)
        Used to calculate the score for each item with the predict_behavior_emb
        """
        return self.item_list_embedding.weight.t()

    def forward(self, interaction):
        item_list_emb = self.item_list_embedding(interaction[self.ITEM_ID_LIST])
        item_list_emb_dropout = self.emb_dropout(item_list_emb)
        short_term_intent_temp, _ = self.gru_layers(item_list_emb_dropout)
        short_term_intent_temp = self.dense(short_term_intent_temp)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        predict_behavior_emb = self.gather_indexes(short_term_intent_temp, interaction[self.ITEM_LIST_LEN] - 1)
        return predict_behavior_emb

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
