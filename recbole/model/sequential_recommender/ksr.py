# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Jin Huang and Shanlei Mu
# @Email  : Betsyj.huang@gmail.com and slmu@ruc.edu.cn


r"""
KSR
################################################

Reference:
# todo 

"""


import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender


class KSR(SequentialRecommender):
    r"""
    # todo 

    Note:
    The sizes of item embedding and entity embedding could be different. For similarity, we make them the same.

    """

    def __init__(self, config, dataset):
        super(KSR, self).__init__(config, dataset)

        # ## get from class KnowledgeRecommender(AbstractRecommender):
        # self.ENTITY_ID = config['ENTITY_ID_FIELD']
        # self.RELATION_ID = config['RELATION_ID_FIELD']
        # self.n_entities = dataset.num(self.ENTITY_ID)
        # self.n_relations = dataset.num(self.RELATION_ID)
        # Todo: Using the above setting causes some errors. So I just use the below fixed setting
        self.n_entities = self.n_items
        self.n_relations = 26
        # load parameters info
        self.device = config['device']

        # # load dataset info
        # self.entity_embedding_matrix = dataset.get_preload_weight('ent_id')
        # self.relation_embedding_matrix = dataset.get_preload_weight('rel_id')
        # Todo: Without relative docs, I randomly generate some data. (See below)
        import numpy as np
        self.embedding_size = config['embedding_size']
        self.entity_embedding_matrix = np.random.randn(self.n_entities, self.embedding_size)
        self.relation_embedding_matrix = np.random.randn(self.n_relations, self.embedding_size)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.gamma = 10 # Todo: Scaling factor

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0) 
        self.entity_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.entity_embedding.weight.requires_grad = False

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
        
        self.loss_fct = BPRLoss()

        # parameters initialization
        self.apply(self._init_weights)
        self.entity_embedding.weight.data.copy_(torch.from_numpy(self.entity_embedding_matrix[:self.n_items]))
        self.relation_Matrix = torch.from_numpy(self.relation_embedding_matrix[:self.n_relations]) # (R+1)*E

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module,nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)

    """ 
        self-defined read and write/update operator:
        memory_update(), sub-function: _memory_update_cell()
        memory_read()
    """
    
    def _get_kg_embedding(self, head):
        # Difference: We generate the embeddings of the tail entities on every relations only for head due to the 1-N problems.
        head_e = self.entity_embedding(head) # B*E
        # # use all relation embs as the relation matrix, M_R, self.relation_embedding
        relation_Matrix = self.relation_Matrix.repeat(head_e.size()[0], 1, 1) # B*R*E
        head_Matrix = torch.unsqueeze(head_e, 1).repeat(1, self.n_relations, 1) # B*R*E
        tail_Matrix = head_Matrix + relation_Matrix
        
        return head_e, tail_Matrix
    
    def _memory_update_cell(self, user_memory, update_memory):
        z = torch.sigmoid(torch.mul(user_memory, update_memory).sum(-1).float()).unsqueeze(-1) # B*R*E -> B*R*1, the gate vector
        # print(z.size(), user_memory.size(), update_memory.size())
        updated_user_memory = (1.0 - z) * user_memory + z * update_memory
        return updated_user_memory

    def memory_update(self, item_seq, item_seq_len): # B*L, B
        step_length = item_seq.size()[1]
        last_item = item_seq_len - 1
        # init user memory with 0s
        user_memory = torch.zeros(item_seq.size()[0], self.n_relations, self.embedding_size).float() # B*R*E
        last_user_memory = torch.zeros_like(user_memory)
        for i in range(step_length): # Loop L
            _, update_memory = self._get_kg_embedding(item_seq[:, i]) # B*R*E
            user_memory = self._memory_update_cell(user_memory, update_memory) # B*R*E
            # print(last_item == i, (last_item==i).sum(), user_memory[last_item==i].size(), last_user_memory[last_item == i].size())
            last_user_memory[last_item == i] = user_memory[last_item == i].float()
        return last_user_memory
    
    def memory_read(self, user_memory, item):
        # Memory Read Operation (attentive combination)
        _, i_memory = self._get_kg_embedding(item) # B*R*E
        attentions = nn.functional.softmax(self.gamma * torch.mul(user_memory, i_memory).sum(-1).float(), -1) # B*R
        u_m = torch.mul(user_memory, attentions.unsqueeze(-1)).sum(1) # B*R*E times B*R*1 -> B*E
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
        gather_index = (item_seq_len - 1).view(-1, 1) # B*1
        last_item = torch.gather(item_seq, 1, gather_index).squeeze() # B*L -> B*1
        u_m = self.memory_read(user_memory, last_item)

        # combine them together
        p_u = self.dense_layer_u(torch.cat((seq_output, u_m), -1))  # B*(2E) -> B*E
        return p_u
    
    def _get_item_comb_embedding(self, item):
        h_e, _ = self._get_kg_embedding(item)
        i_e = self.item_embedding(item)
        q_i = self.dense_layer_i(torch.cat((i_e, h_e), -1))  # B*(2E) -> B*E
        return q_i

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_items_emb = self._get_item_comb_embedding(pos_items)
        neg_items_emb = self._get_item_comb_embedding(neg_items)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1) # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1) # [B]
        loss = self.loss_fct(pos_score, neg_score)
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
        #
        test_items_emb = self.dense_layer_i(torch.cat((self.item_embedding.weight, self.entity_embedding.weight), -1)) # All * E
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
