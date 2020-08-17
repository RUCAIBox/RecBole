# @Time   : 2020/8/17
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn import functional as F
from model.abstract_recommender import SequentialRecommender


class GRU4RecTest(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GRU4RecTest, self).__init__()
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.POSITION_ID = config['POSITION_ID_FIELD']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LEN']
        self.TARGET_ITEM_ID_FIELD = config['TARGET_ITEM_ID_FIELD']
        self.embedding_size = config['embedding_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.batch_size = config['batch_size']
        self.input_length = config['input_length']
        self.item_count = dataset.num(self.ITEM_ID)
        self.position_count = dataset.num(self.POSITION_ID)

        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size)
        self.position_list_embedding = nn.Embedding(self.position_count,self.embedding_size)
        self.gru_layers = nn.GRU(self.embedding_size, self.embedding_size,
                                 self.num_layers,bias=False,batch_first=True, dropout=self.dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.log_softmax = nn.LogSoftmax()
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
        #TODO behavior_list_emb = concat(item,catgory)
        item_list_emb = self.item_list_embedding(interaction[self.ITEM_ID])
        position_list_emb = self.position_list_embedding(interaction[self.POSITION_ID])
        behavior_list_emb = item_list_emb+position_list_emb
        short_term_intent_temp,_ = self.gru_layers(behavior_list_emb)
        short_term_intent_temp = self.gather_indexes(short_term_intent_temp,interaction)
        predict_behavior_emb = self.layer_norm(short_term_intent_temp)
        return predict_behavior_emb

    def gather_indexes(self,gru_output,interaction):
        "Gathers the vectors at the spexific positions over a minibatch"
        mask_index = torch.reshape(interaction[self.ITEM_LIST_LEN]-1,[self.batch_size,1])
        flat_offsets = torch.reshape(torch.arange(self.batch_size)*self.input_length,[-1,1])
        flat_positions = torch.reshape(mask_index+flat_offsets,[-1])
        flat_sequence_tensor = torch.reshape(gru_output,[self.batch_size*self.input_length,self.embedding_size])
        output_tensor = flat_sequence_tensor.index_select(0,flat_positions)
        return output_tensor

    def calculate_loss(self, interaction):
        labels = interaction[self.TARGET_ITEM_ID_FIELD]
        pred = self.forward(interaction)
        logits = torch.matmul(pred,self.get_item_lookup_table())
        log_probs = self.log_softmax(logits)
        labels = torch.reshape(labels,[-1])
        one_hot_labels = F.one_hot(labels,num_classes = self.get_item_lookup_table().shape[0])
        loss_origin = -torch.sum(log_probs.float()*one_hot_labels.float(),dim = -1)
        loss = torch.mean(loss_origin)
        return loss

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        position = interaction[self.POSITION_ID]
        pred = self.forward(interaction)
        scores = torch.matmul(pred, self.get_item_lookup_table())
        return scores
