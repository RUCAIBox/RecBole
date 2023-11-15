# -*- coding: utf-8 -*-
# @Time     : 2020/11/22 8:30
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun, Fan xinyan
# @Email    : shaoweiqi@ruc.edu.cn, xinyan.fan@ruc.edu.cn

r"""
RepeatNet
################################################

Reference:
    Pengjie Ren et al. "RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation."
    in AAAI 2019

Reference code:
    https://github.com/PengjieRen/RepeatNet.

"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class RepeatNet(SequentialRecommender):
    r"""
    RepeatNet explores a hybrid encoder with an repeat module and explore module
    repeat module is used for finding out the repeat consume in sequential recommendation
    explore module is used for exploring new items for recommendation

    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RepeatNet, self).__init__(config, dataset)

        # load the dataset information
        self.device = config["device"]

        # load parameters
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.joint_train = config["joint_train"]
        self.dropout_prob = config["dropout_prob"]

        # define the layers and loss function
        self.item_matrix = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.repeat_explore_mechanism = Repeat_Explore_Mechanism(
            self.device,
            hidden_size=self.hidden_size,
            seq_len=self.max_seq_length,
            dropout_prob=self.dropout_prob,
        )
        self.repeat_recommendation_decoder = Repeat_Recommendation_Decoder(
            self.device,
            hidden_size=self.hidden_size,
            seq_len=self.max_seq_length,
            num_item=self.n_items,
            dropout_prob=self.dropout_prob,
        )
        self.explore_recommendation_decoder = Explore_Recommendation_Decoder(
            hidden_size=self.hidden_size,
            seq_len=self.max_seq_length,
            num_item=self.n_items,
            device=self.device,
            dropout_prob=self.dropout_prob,
        )

        self.loss_fct = F.nll_loss

        # init the weight of the module
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, item_seq, item_seq_len):
        batch_seq_item_embedding = self.item_matrix(item_seq)
        # batch_size * seq_len == embedding ==>> batch_size * seq_len * embedding_size

        all_memory, _ = self.gru(batch_seq_item_embedding)
        last_memory = self.gather_indexes(all_memory, item_seq_len - 1)
        # all_memory: batch_size * item_seq * hidden_size
        # last_memory: batch_size * hidden_size
        timeline_mask = item_seq == 0

        self.repeat_explore = self.repeat_explore_mechanism.forward(
            all_memory=all_memory, last_memory=last_memory
        )
        # batch_size * 2
        repeat_recommendation_decoder = self.repeat_recommendation_decoder.forward(
            all_memory=all_memory,
            last_memory=last_memory,
            item_seq=item_seq,
            mask=timeline_mask,
        )
        # batch_size * num_item
        explore_recommendation_decoder = self.explore_recommendation_decoder.forward(
            all_memory=all_memory,
            last_memory=last_memory,
            item_seq=item_seq,
            mask=timeline_mask,
        )
        # batch_size * num_item
        prediction = repeat_recommendation_decoder * self.repeat_explore[
            :, 0
        ].unsqueeze(1) + explore_recommendation_decoder * self.repeat_explore[
            :, 1
        ].unsqueeze(
            1
        )
        # batch_size * num_item

        return prediction

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_item = interaction[self.POS_ITEM_ID]
        prediction = self.forward(item_seq, item_seq_len)
        loss = self.loss_fct((prediction + 1e-8).log(), pos_item, ignore_index=0)
        if self.joint_train is True:
            loss += self.repeat_explore_loss(item_seq, pos_item)

        return loss

    def repeat_explore_loss(self, item_seq, pos_item):
        batch_size = item_seq.size(0)
        repeat, explore = torch.zeros(batch_size).to(self.device), torch.ones(
            batch_size
        ).to(self.device)
        index = 0
        for seq_item_ex, pos_item_ex in zip(item_seq, pos_item):
            if pos_item_ex in seq_item_ex:
                repeat[index] = 1
                explore[index] = 0
            index += 1
        repeat_loss = torch.mul(
            repeat.unsqueeze(1), torch.log(self.repeat_explore[:, 0] + 1e-8)
        ).mean()
        explore_loss = torch.mul(
            explore.unsqueeze(1), torch.log(self.repeat_explore[:, 1] + 1e-8)
        ).mean()

        return (-repeat_loss - explore_loss) / 2

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        prediction = self.forward(item_seq, item_seq_len)

        return prediction

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        # batch_size * num_items
        seq_output = seq_output.unsqueeze(-1)
        # batch_size * num_items * 1
        scores = self.gather_indexes(seq_output, test_item).squeeze(-1)

        return scores


class Repeat_Explore_Mechanism(nn.Module):
    def __init__(self, device, hidden_size, seq_len, dropout_prob):
        super(Repeat_Explore_Mechanism, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.seq_len = seq_len
        self.Wre = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ure = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vre = nn.Linear(hidden_size, 1, bias=False)
        self.Wcre = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, all_memory, last_memory):
        """
        calculate the probability of Repeat and explore
        """
        all_memory_values = all_memory

        all_memory = self.dropout(self.Ure(all_memory))

        last_memory = self.dropout(self.Wre(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_ere = self.tanh(all_memory + last_memory)

        output_ere = self.Vre(output_ere)
        alpha_are = nn.Softmax(dim=1)(output_ere)
        alpha_are = alpha_are.repeat(1, 1, self.hidden_size)
        output_cre = alpha_are * all_memory_values
        output_cre = output_cre.sum(dim=1)

        output_cre = self.Wcre(output_cre)

        repeat_explore_mechanism = nn.Softmax(dim=-1)(output_cre)

        return repeat_explore_mechanism


class Repeat_Recommendation_Decoder(nn.Module):
    def __init__(self, device, hidden_size, seq_len, num_item, dropout_prob):
        super(Repeat_Recommendation_Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.seq_len = seq_len
        self.num_item = num_item
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vr = nn.Linear(hidden_size, 1)

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        """
        calculate the the force of repeat
        """
        all_memory = self.dropout(self.Ur(all_memory))

        last_memory = self.dropout(self.Wr(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_er = self.tanh(last_memory + all_memory)

        output_er = self.Vr(output_er).squeeze(2)

        if mask is not None:
            output_er.masked_fill_(mask, -1e9)

        output_er = nn.Softmax(dim=-1)(output_er)

        batch_size, b_len = item_seq.size()
        repeat_recommendation_decoder = torch.zeros(
            [batch_size, self.num_item], device=self.device
        )
        repeat_recommendation_decoder.scatter_add_(1, item_seq, output_er)

        return repeat_recommendation_decoder.to(self.device)


class Explore_Recommendation_Decoder(nn.Module):
    def __init__(self, hidden_size, seq_len, num_item, device, dropout_prob):
        super(Explore_Recommendation_Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_item = num_item
        self.device = device
        self.We = nn.Linear(hidden_size, hidden_size)
        self.Ue = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.Ve = nn.Linear(hidden_size, 1)
        self.matrix_for_explore = nn.Linear(
            2 * self.hidden_size, self.num_item, bias=False
        )

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        """
        calculate the force of explore
        """
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.Ue(all_memory))

        last_memory = self.dropout(self.We(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output_ee = self.tanh(all_memory + last_memory)
        output_ee = self.Ve(output_ee).squeeze(-1)

        if mask is not None:
            output_ee.masked_fill_(mask, -1e9)

        output_ee = output_ee.unsqueeze(-1)

        alpha_e = nn.Softmax(dim=1)(output_ee)
        alpha_e = alpha_e.repeat(1, 1, self.hidden_size)
        output_e = (alpha_e * all_memory_values).sum(dim=1)
        output_e = torch.cat([output_e, last_memory_values], dim=1)
        output_e = self.dropout(self.matrix_for_explore(output_e))

        item_seq_first = item_seq[:, 0].unsqueeze(1).expand_as(item_seq)
        item_seq_first = item_seq_first.masked_fill(item_seq > 0, 0)
        item_seq_first.requires_grad_(False)
        output_e.scatter_add_(
            1, item_seq + item_seq_first, float("-inf") * torch.ones_like(item_seq)
        )
        explore_recommendation_decoder = nn.Softmax(1)(output_e)

        return explore_recommendation_decoder
