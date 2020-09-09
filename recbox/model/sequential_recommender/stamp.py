# @Time   : 2020/9/8 19:24
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com


import torch
from torch import nn
from torch.nn.init import normal_
from ...utils import InputType
from ..abstract_recommender import SequentialRecommender


class STAMP(SequentialRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(STAMP, self).__init__()

        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']


        self.embedding_size = config['embedding_size']
        self.item_count = dataset.item_num

        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w0 = nn.Linear(self.embedding_size, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.embedding_size))

        self.mlp_a = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.mlp_b = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.criterion = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def get_item_lookup_table(self):
        return self.item_list_embedding.weight.t()

    def forward(self, interaction):
        item_list_emb = self.item_list_embedding(interaction[self.ITEM_ID_LIST])
        last_inputs = self.gather_indexes(item_list_emb, interaction[self.ITEM_LIST_LEN] - 1)
        org_memory = item_list_emb
        ms = torch.div(torch.sum(org_memory, dim=1), interaction[self.ITEM_LIST_LEN].unsqueeze(1).float())
        alpha = self.count_alpha(org_memory, last_inputs, ms)
        vec = torch.matmul(alpha.unsqueeze(1), org_memory)
        ma = vec.squeeze(1) + ms
        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last_inputs))
        predict_behavior_emb = hs * ht
        return predict_behavior_emb

    def gather_indexes(self, output, gather_index):
        "Gathers the vectors at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, self.embedding_size)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def count_alpha(self, context, aspect, output):
        """
        :param context: org_memory [batch_size, seq_len, emb]
        :param aspect: last_inputs [batch_size, emb]
        :param output: ms [batch_size, emb]
        :return: attention weights
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        output_3dim = output.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha


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
