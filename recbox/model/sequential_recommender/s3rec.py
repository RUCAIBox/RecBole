# -*- coding: utf-8 -*-
# @Time    : 2020/9/19 21:49
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
recbox.model.sequential_recommender.s3rec
################################################

Reference:
Kun Zhou and Hui Wang et al. "S^3-Rec: Self-Supervised Learning
for Sequential Recommendation with Mutual Information Maximization"
In CIKM 2020.

The authors' implementation
https://github.com/RUCAIBox/CIKM2020-S3Rec

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

class S3Rec(SequentialRecommender):
    r"""
    S3Rec is the first work to incorporate self-supervised learning in
    sequential recommendation.

    NOTE: Under this framework, we need reconstruct the pretraining data,
    which would affect the pre-training speed.
    """
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(S3Rec, self).__init__()

        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.FEATURE_FIELD = config['FEATURE_FIELD']
        self.FEATURE_LIST = self.FEATURE_FIELD + config['LIST_SUFFIX']

        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_count = dataset.item_num + 1 # for mask token
        self.feature_count = dataset.num(self.FEATURE_FIELD) - 1  # we don't need padding
        self.item_feat = dataset.get_item_feature()
        ## training config
        self.train_stage = config['train_stage']  # pretrain or finetune
        self.pre_model_path = config['pre_model_path']  # We need this for finetune

        ## modules shared by pre-training stage and fine-tuning stage
        self.item_embedding = nn.Embedding(self.item_count, config['hidden_size'], padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_item_list_length, config['hidden_size'], padding_idx=0)
        self.feature_embedding = nn.Embedding(self.feature_count, config['hidden_size'])
        self.trm_encoder = TransformerEncoder(config)

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['dropout_prob'])

        # for pretrain
        self.mask_token = self.item_count - 1
        self.mask_ratio = config['mask_ratio']
        self.aap_weight = config['aap_weight']
        self.mip_weight = config['mip_weight']
        self.map_weight = config['map_weight']
        self.sp_weight = config['sp_weight']

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.mip_norm = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.map_norm = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.sp_norm = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.criterion = nn.BCELoss(reduction='none')


        # For finetune

        self.loss_type = config['loss_type']
        self.bpr_loss = BPRLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.initializer_range = config['initializer_range']


        if config['train_stage'] == 'pretrain':
            self.apply(self._init_weights)
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            print('Load pretrained model from', self.pre_model_path)
            self.load_state_dict(pretrained['state_dict'])

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

    # AAP
    def associated_attribute_prediction(self, sequence_output, feature_embedding):
        '''
        :param sequence_output: [B L H]
        :param feature_embedding: [feature_num H]
        :return: scores [B*L feature_num]
        '''
        sequence_output = self.aap_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, sequence_output.size(-1), 1])  # [B*L H 1]
        # [feature_num H] [B*L H 1] -> [B*L feature_num 1]
        score = torch.matmul(feature_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L feature_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item_emb):
        '''
        :param sequence_output: [B L H]
        :param target_item_emb: [B L H]
        :return: scores [B*L]
        '''
        sequence_output = self.mip_norm(sequence_output.view([-1, sequence_output.size(-1)]))  # [B*L H]
        target_item_emb = target_item_emb.view([-1, sequence_output.size(-1)])  # [B*L H]
        score = torch.mul(sequence_output, target_item_emb)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, sequence_output.size(-1), 1])  # [B*L H 1]
        # [feature_num H] [B*L H 1] -> [B*L feature_num 1]
        score = torch.matmul(feature_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L feature_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment_emb):
        '''
        :param context: [B H]
        :param segment_emb: [B H]
        :return:
        '''
        context = self.sp_norm(context)
        score = torch.mul(context, segment_emb)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    def get_attention_mask(self, sequence, bidirectional=True):
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        if not bidirectional:
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(sequence.device)
            extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def encode_sequence(self, sequence, bidirectional=True):
        position_ids = torch.arange(sequence.size(1), dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(sequence)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        attention_mask = self.get_attention_mask(sequence, bidirectional=bidirectional)
        trm_output = self.trm_encoder(input_emb,
                                      attention_mask,
                                      output_all_encoded_layers=True)
        output = trm_output[-1] # [B L H]
        return output

    def pretrain(self, features, masked_item_sequence, pos_items, neg_items,
                 masked_segment_sequence, pos_segment, neg_segment):

        # Encode masked sequence
        sequence_output = self.encode_sequence(masked_item_sequence)

        feature_embedding = self.feature_embedding.weight
        # AAP
        aap_score = self.associated_attribute_prediction(sequence_output, feature_embedding)
        aap_loss = self.criterion(aap_score, features.view(-1, self.feature_count).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.mask_token).float() * \
                   (masked_item_sequence != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_sequence == self.mask_token).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self.masked_attribute_prediction(sequence_output, feature_embedding)
        map_loss = self.criterion(map_score, features.view(-1, self.feature_count).float())
        map_mask = (masked_item_sequence == self.mask_token).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        # take the last position hidden as the context
        segment_context = self.encode_sequence(masked_segment_sequence)[:, -1, :]  # [B H]
        pos_segment_emb = self.encode_sequence(pos_segment)[:, -1, :]
        neg_segment_emb = self.encode_sequence(neg_segment)[:, -1, :]  # [B H]
        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)
        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)
        sp_loss = torch.sum(self.criterion(sp_distance,
                                           torch.ones_like(sp_distance, dtype=torch.float32)))

        pretrain_loss = self.aap_weight*aap_loss \
                        + self.mip_weight*mip_loss \
                        + self.map_weight*map_loss \
                        + self.sp_weight*sp_loss

        return pretrain_loss

    def finetune(self, interaction):
        sequence = interaction[self.ITEM_ID_LIST]
        # we use uni-directional attention in the fine-tuning stage
        seq_output = self.encode_sequence(sequence, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, interaction[self.ITEM_LIST_LEN] - 1)
        pos_items = interaction[self.TARGET_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)  # [B H]
            neg_items_emb = self.item_embedding(neg_items)  # [B H]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.bpr_loss(pos_score, neg_score)
            return loss
        elif self.loss_type == 'CE':
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.ce_loss(logits, pos_items)
            return loss
        else:
            raise NotImplementedError

    def gather_indexes(self, output, gather_index):
        "Gathers the vectors at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.size(-1))
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def neg_sample(self, item_set):  # [ , ]
        item = random.randint(1, self.item_count - 1)
        while item in item_set:
            item = random.randint(1, self.item_count - 1)
        return item

    def padding_zero_at_left(self, sequence):
        # had truncated according to the max_length
        pad_len = self.max_item_list_length - len(sequence)
        sequence = [0]*pad_len + sequence
        return sequence

    # the data could be directly used in fine-tune stage
    def reconstruct_pretrain_data(self, interaction):

        item_list = interaction[self.ITEM_ID_LIST]
        device = item_list.device
        batch_size = item_list.size(0)

        # We don't need padding for features
        item_feature_list = self.item_feat[self.FEATURE_FIELD][item_list] - 1
        
        end_index = interaction[self.ITEM_LIST_LEN].cpu().numpy().tolist()
        item_list = item_list.cpu().numpy().tolist()
        item_feature_list = item_feature_list.cpu().numpy().tolist()

        # we will padding zeros at the left side
        # these will be train_instances, after will be reshaped to batch
        sequence_instances = []
        associated_features = [] # For Associated Attribute Prediction and Masked Attribute Prediction
        long_sequence = []
        for i, end_i in enumerate(end_index):
            sequence_instances.append(item_list[i][:end_i])
            long_sequence.extend(item_list[i][:end_i])
            # padding feature at the left side
            associated_features.extend([[0] * self.feature_count] * (self.max_item_list_length - end_i))
            for indexes in item_feature_list[i][:end_i]:
                features = [0] * self.feature_count
                try:
                    # multi class
                    for index in indexes:
                        if index >= 0:
                            features[index] = 1
                except:
                    # single class
                    features[indexes] = 1
                associated_features.append(features)

        # Masked Item Prediction and Masked Attribute Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        for instance in sequence_instances:
            masked_sequence = instance.copy()
            pos_item = instance.copy()
            neg_item = instance.copy()
            for index_id, item in enumerate(instance):
                prob = random.random()
                if prob < self.mask_ratio:
                    masked_sequence[index_id] = self.mask_token
                    neg_item[index_id] = self.neg_sample(instance)
            masked_item_sequence.append(self.padding_zero_at_left(masked_sequence))
            pos_items.append(self.padding_zero_at_left(pos_item))
            neg_items.append(self.padding_zero_at_left(neg_item))

        # Segment Prediction
        masked_segment_list = []
        pos_segment_list = []
        neg_segment_list = []
        for instance in sequence_instances:
            if len(instance) < 2:
                masked_segment = instance.copy()
                pos_segment = instance.copy()
                neg_segment = instance.copy()
            else:
                sample_length = random.randint(1, len(instance) // 2)
                start_id = random.randint(0, len(instance) - sample_length)
                neg_start_id = random.randint(0, len(long_sequence) - sample_length)
                pos_segment = instance[start_id: start_id + sample_length]
                neg_segment = long_sequence[neg_start_id:neg_start_id + sample_length]
                masked_segment = instance[:start_id] + [self.mask_token] * sample_length \
                                 + instance[start_id + sample_length:]
                pos_segment = [self.mask_token] * start_id + pos_segment + [self.mask_token] * (
                        len(instance) - (start_id + sample_length))
                neg_segment = [self.mask_token] * start_id + neg_segment + [self.mask_token] * (
                        len(instance) - (start_id + sample_length))
            masked_segment_list.append(self.padding_zero_at_left(masked_segment))
            pos_segment_list.append(self.padding_zero_at_left(pos_segment))
            neg_segment_list.append(self.padding_zero_at_left(neg_segment))

        associated_features = torch.tensor(associated_features, dtype=torch.long, device=device)
        associated_features = associated_features.view(-1, self.max_item_list_length, self.feature_count)

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        masked_segment_list = torch.tensor(masked_segment_list, dtype=torch.long, device=device).view(batch_size, -1)
        pos_segment_list = torch.tensor(pos_segment_list, dtype=torch.long, device=device).view(batch_size, -1)
        neg_segment_list = torch.tensor(neg_segment_list, dtype=torch.long, device=device).view(batch_size, -1)

        return associated_features, masked_item_sequence, pos_items, neg_items, \
               masked_segment_list, pos_segment_list, neg_segment_list


    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            features, masked_item_sequence, pos_items, neg_items, \
            masked_segment_sequence, pos_segment, neg_segment \
                = self.reconstruct_pretrain_data(interaction)

            loss = self.pretrain(features, masked_item_sequence, pos_items, neg_items,
                                 masked_segment_sequence, pos_segment, neg_segment)
        else:
            loss = self.finetune(interaction)
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        sequence = interaction[self.ITEM_ID_LIST]
        seq_output = self.encode_sequence(sequence, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, interaction[self.ITEM_LIST_LEN] - 1)
        test_item_emb = self.item_embedding.weight[:self.item_count-1] # delete masked token
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, item_num]
        return scores