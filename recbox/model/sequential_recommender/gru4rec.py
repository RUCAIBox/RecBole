# @Time   : 2020/8/17 19:38
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE:
# @Time   : 2020/8/26 17:16
# @Author : Yupeng Hou,Yujie Lu
# @Email  : houyupeng@ruc.edu.cn,yujielu1998@gmail.com

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from ...utils import InputType
from ..abstract_recommender import SequentialRecommender


class GRU4Rec(SequentialRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(GRU4Rec, self).__init__()

        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID


        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.item_count = dataset.item_num

        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size,padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size,self.embedding_size)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_uniform_(module.weight)
        elif isinstance(module,nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)

    def get_item_lookup_table(self):
        return self.item_list_embedding.weight.t()

    def forward(self, interaction):
        item_list_emb = self.item_list_embedding(interaction[self.ITEM_ID_LIST])
        item_list_emb_dropout = self.emb_dropout(item_list_emb)
        short_term_intent_temp, _ = self.gru_layers(item_list_emb_dropout)
        short_term_intent_temp = self.dense(short_term_intent_temp)
        predict_behavior_emb = self.gather_indexes(short_term_intent_temp, interaction[self.ITEM_LIST_LEN] - 1)
        return predict_behavior_emb

    def gather_indexes(self, gru_output, gather_index):
        "Gathers the vectors at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, self.embedding_size)
        output_tensor = gru_output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

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
