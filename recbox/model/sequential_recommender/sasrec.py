# @Time   : 2020/8/24 12:18
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/9/10
# @Author : Yupeng Hou
# @email  : houyupeng@ruc.edu.cn

import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from ...utils import InputType
from ..abstract_recommender import SequentialRecommender
from ..layers import MultiHeadAttention


class SASRec(SequentialRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(SASRec, self).__init__()

        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.POSITION_ID = config['POSITION_FIELD']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID
        self.device = config['device']
        max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.embedding_size = config['embedding_size']
        self.n_head = config['n_head']
        self.d_model = self.embedding_size
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']
        self.num_blocks = config['num_blocks']
        self.item_count = dataset.item_num
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_model // self.n_head

        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)
        self.position_list_embedding = nn.Embedding(max_item_list_length, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.dropout[0])
        self.multi_head_attention = MultiHeadAttention(self.n_head, self.d_model, self.d_k, self.d_v)
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.conv1d_1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_ff, kernel_size=1, bias=True)
        self.conv1d_2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.d_model, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.conv1_dropout = nn.Dropout(p=self.dropout[1])
        self.conv2_dropout = nn.Dropout(p=self.dropout[2])
        self.criterion = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)

    def get_item_lookup_table(self):
        return self.item_list_embedding.weight.t()

    def forward(self, interaction):
        item_list_emb = self.item_list_embedding(interaction[self.ITEM_ID_LIST])
        position_list_emb = self.position_list_embedding(interaction[self.POSITION_ID])
        behavior_list_emb = item_list_emb + position_list_emb
        behavior_list_emb_drop = self.emb_dropout(behavior_list_emb)
        key_padding_mask = self.get_attn_pad_mask(interaction[self.ITEM_ID_LIST], interaction[self.ITEM_ID_LIST])
        look_ahead_mask = self.get_attn_subsequence_mask(interaction[self.ITEM_ID_LIST])
        mask = torch.gt((key_padding_mask + look_ahead_mask), 0)
        attn_weights = []
        attn_outputs = behavior_list_emb_drop
        for i in range(self.num_blocks):
            attn_outputs, attn = self.multi_head_attention(attn_outputs, attn_outputs, attn_outputs, mask)
            attn_weights.append(attn)
            attn_outputs = self.feedforward(attn_outputs)
        long_term_prefernce = self.gather_indexes(attn_outputs, interaction[self.ITEM_LIST_LEN] - 1)
        predict_behavior_emb = self.layer_norm(long_term_prefernce)
        return predict_behavior_emb, attn_weights

    def get_attn_pad_mask(self, seq_q, seq_k):
        '''
            seq_q: [batch_size, seq_len]
            seq_k: [batch_size, seq_len]
            seq_len could be src_len or it could be tgt_len
            seq_len in seq_q and seq_len in seq_k maybe not equal
        '''
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequence_mask(self, seq):
        '''
            seq: [batch_size, tgt_len]
        '''
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        ones = torch.ones(attn_shape, dtype=torch.uint8, device=self.device)
        subsequence_mask = ones.triu(diagonal=1)
        return subsequence_mask

    def feedforward(self, x):
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv1d_1(x)
        x = self.relu(x)
        x = self.conv1_dropout(x)
        x = self.conv1d_2(x)
        x = x.permute(0, 2, 1)
        x = self.conv2_dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x

    def calculate_loss(self, interaction):
        target_id = interaction[self.TARGET_ITEM_ID]
        pred, _ = self.forward(interaction)
        logits = torch.matmul(pred, self.get_item_lookup_table())
        loss = self.criterion(logits, target_id)
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        pred,_ = self.forward(interaction)
        scores = torch.matmul(pred, self.get_item_lookup_table())
        return scores
