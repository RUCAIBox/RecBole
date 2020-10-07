# -*- coding: utf-8 -*-
# @Time   : 2020/8/25 19:56
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/9/15, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com
r"""
recbox.model.sequential_recommender.narm
################################################

Reference:
Jing Li et al. "Neural Attentive Session-based Recommendation." in CIKM 2017.

Reference code:
https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch

"""
import torch
from torch import nn
from recbox.utils import InputType
from recbox.model.abstract_recommender import SequentialRecommender
from torch.nn.init import xavier_normal_, constant_


class NARM(SequentialRecommender):
    r"""NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
    and capture the user’s main purpose in the current session.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NARM, self).__init__()
        # load parameters info
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.device = config['device']
        self.item_count = dataset.item_num

        # item embeddings
        self.item_list_embedding = nn.Embedding(self.item_count,
                                                self.embedding_size,
                                                padding_idx=0)
        # define layers and loss
        self.emb_dropout = nn.Dropout(self.dropout[0])
        self.gru = nn.GRU(self.embedding_size,
                          self.hidden_size,
                          self.n_layers,
                          bias=False,
                          batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout[1])
        self.b = nn.Linear(2 * self.hidden_size,
                           self.embedding_size,
                           bias=False)
        self.criterion = nn.CrossEntropyLoss()
        # parameters initialization
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_item_lookup_table(self):
        r"""Get the transpose of item_list_embedding.weight，Shape of (embedding_size, item_count+padding_id)
        Used to calculate the score for each item with the predict_behavior_emb
        """
        return self.item_list_embedding.weight.t()

    def forward(self, interaction):
        item_id_list = interaction[self.ITEM_ID_LIST]
        item_list_emb = self.item_list_embedding(item_id_list)
        item_list_emb_dropout = self.emb_dropout(item_list_emb)
        gru_out, _ = self.gru(item_list_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(
            gru_out, interaction[self.ITEM_LIST_LEN] - 1)
        # avoid the influence of padding
        mask = item_id_list.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        pred = self.b(c_t)

        return pred

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
