# @Time   : 2020/8/25 19:56
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com


import torch
from torch import nn
from ...utils import InputType
from ..abstract_recommender import SequentialRecommender
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


# TODO:init
class NARM(SequentialRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(NARM, self).__init__()

        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.item_count = dataset.item_num


        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout[0])
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.n_layers, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout[1])
        self.b = nn.Linear(2*self.hidden_size, self.embedding_size, bias=False)
        self.criterion = nn.CrossEntropyLoss()



    def get_item_lookup_table(self):
        return self.item_list_embedding.weight.t()

    def forward(self, interaction):
        item_id_list = interaction[self.ITEM_ID_LIST]
        index = interaction[self.ITEM_LIST_LEN]
        index = index.view(item_id_list.size(0), 1)
        #reset masked_id to 0
        item_id_list.scatter_(dim=1, index=index, src=torch.zeros_like(item_id_list))
        item_list_emb = self.item_list_embedding(item_id_list)
        item_list_emb_dropout = self.emb_dropout(item_list_emb)
        item_list_emb_nopad = pack_padded_sequence(
            input=item_list_emb_dropout,
            lengths=interaction[self.ITEM_LIST_LEN],
            batch_first=True,
            enforce_sorted=False)
        gru_out, hidden = self.gru(item_list_emb_nopad)
        gru_out, lengths = pad_packed_sequence(gru_out, batch_first=True, total_length=self.max_item_list_length)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        c_global = ht

        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)

        mask = torch.where(item_id_list>0, torch.tensor([1.]), torch.tensor([0.]))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1)*q2_expand

        alpha = self.v_t(torch.sigmoid(q1+q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out)*gru_out, 1)

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
