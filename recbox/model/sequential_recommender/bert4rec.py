# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 12:08
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
recbox.model.sequential_recommender.bert4rec
################################################

Reference:
Fei Sun et al. "BERT4Rec: Sequential Recommendation with
Bidirectional Encoder Representations from Transformer."
In CIKM 2019.

Reference:
The authors' tensorflow implementation
https://github.com/FeiSun/BERT4Rec

"""

import random

import torch
import time
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbox.utils import InputType
from recbox.model.abstract_recommender import SequentialRecommender
from recbox.model.loss import BPRLoss
from recbox.model.init import xavier_normal_initialization
from recbox.model.layers import TransformerEncoder

class BERT4Rec(SequentialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BERT4Rec, self).__init__()
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID

        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_count = dataset.item_num + 1 # for mask token
        self.mask_token = self.item_count - 1
        self.mask_ratio = config['mask_ratio']

        self.item_embedding = nn.Embedding(self.item_count, config['hidden_size'], padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_item_list_length, config['hidden_size'], padding_idx=0)

        self.trm_encoder = TransformerEncoder(config)
        # For input
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['dropout_prob'])

        self.loss_type = config['loss_type']
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') # we only need compute the loss at the masked position
        self.mask_item_length = int(self.mask_ratio * self.max_item_list_length)
        self.initializer_range = config['initializer_range']
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        "Gathers the vectors at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.size(-1))
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_list):
        attention_mask = (item_list > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def neg_sample(self, item_set):
        item = random.randint(1, self.item_count - 1)
        while item in item_set:
            item = random.randint(1, self.item_count - 1)
        return item

    def padding_zero_at_left(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0]*pad_len + sequence
        sequence = sequence[-max_length:] # truncate according to the max_length
        return sequence

    # mask data for training
    # 0.1s/batch for reconstruction
    def reconstruct_train_data(self, interaction):
        # concat target at the last position, but the last position is not at the 'last'
        item_list = interaction[self.ITEM_ID_LIST]
        device = item_list.device
        batch_size = item_list.size(0)

        targets = interaction[self.TARGET_ITEM_ID].cpu().numpy().tolist()
        end_index = interaction[self.ITEM_LIST_LEN]
        item_list = item_list.cpu().numpy().tolist()

        # we will padding zeros at the left side
        # these will be train_instances, after will be reshaped to batch
        sequence_instances = []
        for i, (end_i, target) in enumerate(zip(end_index, targets)):
            instance = item_list[i][:end_i]
            instance.append(target)
            sequence_instances.append(instance)

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []
        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            neg_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    neg_item.append(self.neg_sample(instance))
                    masked_sequence[index_id] = self.mask_token
                    # padding is zero, we will -1 later
                    index_ids.append(index_id+1)

            masked_item_sequence.append(self.padding_zero_at_left(masked_sequence, self.max_item_list_length))
            pos_items.append(self.padding_zero_at_left(pos_item, self.mask_item_length))
            neg_items.append(self.padding_zero_at_left(neg_item, self.mask_item_length))
            masked_index.append(self.padding_zero_at_left(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)

        return masked_item_sequence, pos_items, neg_items, masked_index

    # we need add mask_token at the last position according to the lengths of item_list
    def reconstruct_test_data(self, interaction):

        # concat target at the last position, but the last position is not at the 'last'
        item_list = interaction[self.ITEM_ID_LIST]
        padding = torch.zeros(item_list.size(0), dtype=torch.long, device=item_list.device)  # [B]
        item_list = torch.cat((item_list, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(interaction[self.ITEM_LIST_LEN]):
            item_list[batch_id][last_position] = self.mask_token
        return item_list

    def forward(self, item_list):

        position_ids = torch.arange(item_list.size(1), dtype=torch.long, device=item_list.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_list)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_list)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_list)
        trm_output = self.trm_encoder(input_emb,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True)
        output = trm_output[-1]
        return output  # [B L H]

    # return a multi-hot vector for every masked sequence to
    # gather the masked position hidden representation
    def multi_hot_embed(self, masked_index, max_length):
        '''
        :param lables: [B mask_len]
        :param max_length:
        :return: [B mask_len max_length]
        '''
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length).cuda()
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        masked_item_list, pos_items, neg_items, masked_index = self.reconstruct_train_data(interaction)
        seq_output = self.forward(masked_item_list)
        # we add 1 to the index before.
        pred_index_map = self.multi_hot_embed(masked_index - 1, masked_item_list.size(-1))
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)
        # [B max_len H] -> [B mask_len H]
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        if self.loss_type == 'BPR':
            pos_items_emb = self.item_embedding(pos_items)  # [B mask_len H]
            neg_items_emb = self.item_embedding(neg_items)  # [B mask_len H]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B mask_len]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B mask_len]
            targets = (masked_index > 0).float()
            # only calculate loss for masked position
            loss = - torch.sum(torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets) \
                   / torch.sum(targets)
            return loss

        elif self.loss_type == 'CE':
            test_item_emb = self.item_embedding.weight # [num H]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # [B mask_len item_num]
            targets = (masked_index > 0).float().view(-1) # [B*mask_len]
            loss = torch.sum(self.ce_loss(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
                   / torch.sum(targets)
            return loss
        else:
            raise NotImplementedError

    # TODO implemented after the data interface is ready
    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        item_list = self.reconstruct_test_data(interaction)
        seq_output = self.forward(item_list)
        seq_output = self.gather_indexes(seq_output, interaction[self.ITEM_LIST_LEN]) # [B H]
        test_item_emb = self.item_embedding.weight[:self.item_count-1] # delete masked token
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, item_num]
        return scores