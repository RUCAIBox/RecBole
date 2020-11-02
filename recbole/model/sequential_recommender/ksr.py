# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Jin Huang and Shanlei Mu
# @Email  : Betsyj.huang@gmail.com and slmu@ruc.edu.cn


r"""
KSR
################################################

Reference:
    Jin Huang et al. "Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks."
    In SIGIR 2018

"""


import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender


class KSR(SequentialRecommender):
    r"""
    KSR integrates the RNN-based networks with Key-Value Memory Network (KV-MN).
    And it further incorporates knowledge base (KB) information to enhance the semantic representation of KV-MN.

    """

    def __init__(self, config, dataset):
        super(KSR, self).__init__(config, dataset)

        # load dataset info
        self.ENTITY_ID = config['ENTITY_ID_FIELD']
        self.RELATION_ID = config['RELATION_ID_FIELD']
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID) - 1
        self.entity_embedding_matrix = dataset.get_preload_weight('ent_id')
        self.relation_embedding_matrix = dataset.get_preload_weight('rel_id')

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.gamma = config['gamma'] # Scaling factor
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.freeze_kg = config['freeze_kg']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0) 
        self.entity_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.entity_embedding.weight.requires_grad = not self.freeze_kg

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        self.dense_layer_u = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.dense_layer_i = nn.Linear(self.embedding_size * 2, self.embedding_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # parameters initialization
        self.apply(self._init_weights)
        self.entity_embedding.weight.data.copy_(torch.from_numpy(self.entity_embedding_matrix[:self.n_items]))
        self.relation_Matrix = torch.from_numpy(self.relation_embedding_matrix[:self.n_relations]).to(self.device) # [R H]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module,nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)
    
    def _get_kg_embedding(self, head):
        """Difference: We generate the embeddings of the tail entities on every relations only for head due to the 1-N problems. """
        head_e = self.entity_embedding(head) # [B H]
        relation_Matrix = self.relation_Matrix.repeat(head_e.size()[0], 1, 1) # [B R H]
        head_Matrix = torch.unsqueeze(head_e, 1).repeat(1, self.n_relations, 1) # [B R H]
        tail_Matrix = head_Matrix + relation_Matrix
        
        return head_e, tail_Matrix
    
    def _memory_update_cell(self, user_memory, update_memory):
        z = torch.sigmoid(torch.mul(user_memory, update_memory).sum(-1).float()).unsqueeze(-1) # [B R 1], the gate vector
        updated_user_memory = (1.0 - z) * user_memory + z * update_memory
        return updated_user_memory

    def memory_update(self, item_seq, item_seq_len):
        """ define write operator """
        step_length = item_seq.size()[1]
        last_item = item_seq_len - 1
        # init user memory with 0s
        user_memory = torch.zeros(item_seq.size()[0], self.n_relations, self.embedding_size).float().to(self.device) # [B R H]
        last_user_memory = torch.zeros_like(user_memory)
        for i in range(step_length): # [len]
            _, update_memory = self._get_kg_embedding(item_seq[:, i]) # [B R H]
            user_memory = self._memory_update_cell(user_memory, update_memory) # [B R H]
            last_user_memory[last_item == i] = user_memory[last_item == i].float()
        return last_user_memory
    
    def memory_read(self, user_memory):
        """ define read operator """
        attrs = self.relation_Matrix
        attentions = nn.functional.softmax(self.gamma * torch.mul(user_memory, attrs).sum(-1).float(), -1) # [B R]
        u_m = torch.mul(user_memory, attentions.unsqueeze(-1)).sum(1) # [B H]
        return u_m

    def forward(self, item_seq, item_seq_len):
        # sequential preference h^u_t
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)

        # attribute-based preference representation, m^u_t
        user_memory = self.memory_update(item_seq, item_seq_len)
        # gather_index = (item_seq_len - 1).view(-1, 1) # [B 1]
        # last_item = torch.gather(item_seq, 1, gather_index).squeeze() # [B 1]
        u_m = self.memory_read(user_memory)

        # combine them together
        p_u = self.dense_layer_u(torch.cat((seq_output, u_m), -1))  # [B H]
        return p_u
    
    def _get_item_comb_embedding(self, item):
        h_e, _ = self._get_kg_embedding(item)
        i_e = self.item_embedding(item)
        q_i = self.dense_layer_i(torch.cat((i_e, h_e), -1))  # [B H]
        return q_i

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self._get_item_comb_embedding(pos_items)
            neg_items_emb = self._get_item_comb_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1) # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1) # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else: # self.loss_type = 'CE'
            test_items_emb = self.dense_layer_i(torch.cat((self.item_embedding.weight, self.entity_embedding.weight), -1)) # [n_items H]
            logits = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) 
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self._get_item_comb_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.dense_layer_i(torch.cat((self.item_embedding.weight, self.entity_embedding.weight), -1)) # [n_items H]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
