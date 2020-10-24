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

from recbole.utils import InputType
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization


class GRU4RecKG(SequentialRecommender):
    r"""It is an extension of GRU4Rec, which concatenates item and its corresponding
    pre-trained knowledge graph embedding feature as the input.

    """
    input_type = InputType.POINTWISE

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
        self.ce_loss = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.entity_embedding.weight.data.copy_(torch.from_numpy(self.entity_embedding_matrix[:self.n_items]))

    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_emb = self.item_embedding(item_seq)
        entity_emb = self.entity_embedding(item_seq)
        item_emb = nn.Dropout(self.dropout)(item_emb)
        entity_emb = nn.Dropout(self.dropout)(entity_emb)

        item_gru_output, _ = self.item_gru_layers(item_emb)  # [B Len H]
        entity_gru_output, _ = self.entity_gru_layers(entity_emb)

        output_concat = torch.cat((item_gru_output, entity_gru_output), -1)  # [B Len 2*H]
        output = self.dense_layer(output_concat)
        output = self.gather_indexes(output, interaction[self.ITEM_SEQ_LEN] - 1)  # [B H]
        return output

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction)
        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.ce_loss(logits, pos_items)
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))    # [B, item_num]
        return scores
