# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:32
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""

recbox.model.sequential_recommender.sasrecf
################################################

This is an extension of SASRec, which concatenates
item representations and item attribute representations as the input
to the model.

"""

import torch
from torch import nn

from recbox.utils import InputType
from recbox.model.abstract_recommender import SequentialRecommender
from recbox.model.loss import BPRLoss
from recbox.model.layers import TransformerEncoder, FeatureSeqEmbLayer



class SASRecF(SequentialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SASRecF, self).__init__()

        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID

        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_count = dataset.item_num

        # need change the 'load_col' config
        self.hidden_size = config['hidden_size']
        self.embedding_size = self.hidden_size

        # embedding_size is same as hidden_size
        self.item_embedding = nn.Embedding(self.item_count, config['hidden_size'], padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_item_list_length, config['hidden_size'], padding_idx=0)
        self.feature_embed_layer = FeatureSeqEmbLayer(config, dataset)

        # For simplicity, we use same architecture for item_trm and feature_trm
        self.trm_encoder = TransformerEncoder(config)

        # For input
        self.num_feature_field = len(config['selected_features'])

        self.concat_layer = nn.Linear(config['hidden_size'] * (1 + self.num_feature_field), config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['dropout_prob'])

        self.loss_type = config['loss_type']
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

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
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_list.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, interaction):
        item_seq = interaction[self.ITEM_ID_LIST]
        item_emb = self.item_embedding(item_seq)

        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']

        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        feature_table = torch.cat(feature_table, dim=1)
        table_shape = feature_table.shape

        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(table_shape[:-2] + (feat_num * embedding_size,))

        input_concat = torch.cat((item_emb, feature_emb), -1)  # [B 1+field_num*H]
        # TODO whether need layer_norm drouout
        input_emb = self.concat_layer(input_concat)

        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask,
                                      output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, interaction[self.ITEM_LIST_LEN] - 1)
        return output # [B H]

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction)
        pos_items = interaction[self.TARGET_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items) # [B H]
            neg_items_emb = self.item_embedding(neg_items) # [B H]
            pos_score = torch.sum(seq_output*pos_items_emb, dim=-1) # [B]
            neg_score = torch.sum(seq_output*neg_items_emb, dim=-1) # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        elif self.loss_type == 'CE':
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    # TODO implemented after the data interface is ready
    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # [B, item_num]
        return scores