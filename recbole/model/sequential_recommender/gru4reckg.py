# @Time   : 2020/10/10
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn


r"""
GRU4RecKG
################################################
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss


class GRU4RecKG(SequentialRecommender):
    r"""It is an extension of GRU4Rec, which concatenates item and its corresponding
    pre-trained knowledge graph embedding feature as the input.

    """

    def __init__(self, config, dataset):
        super(GRU4RecKG, self).__init__(config, dataset)

        # load dataset info
        self.entity_embedding_matrix = dataset.get_preload_weight('ent_id')

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout_prob']
        self.freeze_kg = config['freeze_kg']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.entity_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.entity_embedding.weight.requires_grad = not self.freeze_kg
        self.item_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.entity_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense_layer = nn.Linear(self.hidden_size * 2, self.embedding_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.entity_embedding.weight.data.copy_(torch.from_numpy(self.entity_embedding_matrix[:self.n_items]))

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        entity_emb = self.entity_embedding(item_seq)
        item_emb = nn.Dropout(self.dropout)(item_emb)
        entity_emb = nn.Dropout(self.dropout)(entity_emb)

        item_gru_output, _ = self.item_gru_layers(item_emb)  # [B Len H]
        entity_gru_output, _ = self.entity_gru_layers(entity_emb)

        output_concat = torch.cat((item_gru_output, entity_gru_output), -1)  # [B Len 2*H]
        output = self.dense_layer(output_concat)
        output = self.gather_indexes(output, item_seq_len - 1)  # [B H]
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)  # [B H]
            neg_items_emb = self.item_embedding(neg_items)  # [B H]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
