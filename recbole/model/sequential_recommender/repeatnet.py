# -*- coding: utf-8 -*-
# @Time    : 2020/11/22 8:30
# @Author  : Shao Weiqi
# @Email   : shaoweiqi@ruc.edu.cn

r"""
RepeatNet
################################################

Reference:
    Pengjie Ren et al. "RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation." in AAAI 2019


"""


import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_

from torch.nn import functional as F
from recbole.model.abstract_recommender import SequentialRecommender


class RepeatNet(SequentialRecommender):
    r"""
    RepeatNet explores a hybrid encoder with an repeat module and explore module
    repeat module is used for finding out the repeat consume in sequential recommendation
    explore module is used for exploring new items for recommendation

    """

    def __init__(self,config,dataset):

        super(RepeatNet, self).__init__(config,dataset)

        # load the dataset information
        self.device=config["device"]

        # load parameters
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.joint_train=config["joint_train"]
        self.dropout_prob=config["dropout_prob"]

        # define the layers and loss function
        self.item_matrix = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.repeat_explore_mechanism = Repeat_Explore_Mechanism(hidden_size=self.hidden_size,
                                                                 seq_len=self.max_seq_length,
                                                                 dropout_prob=self.dropout_prob)
        self.repeat_recommendation_decoder = Repeat_Recommendation_Decoder(hidden_size=self.hidden_size,
                                                                           seq_len=self.max_seq_length,
                                                                           num_item=self.n_items,
                                                                           dropout_prob=self.dropout_prob)
        self.explore_recommendation_decoder = Explore_Recommendation_Decoder(hidden_size=self.hidden_size,
                                                                             seq_len=self.max_seq_length,
                                                                             num_item=self.n_items,
                                                                             dropout_prob=self.dropout_prob)

        self.loss_fct=F.nll_loss

        # init the weight of the module
        self.apply(self._init_weights)


    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def forward(self, seq_item):

        batch_seq_item_embedding = self.item_matrix(seq_item)
        # batch_size * seq_len == embedding ==>> batch_size * seq_len * embedding_size

        all_memory, last_memory = self.gru(batch_seq_item_embedding)
        last_memory = last_memory.squeeze(0)
        # all_memory: batch_size * seq_item * hidden_size
        # last_memory: batch_size * hidden_size

        repeat_recommendation_mechanism = self.repeat_explore_mechanism.forward(all_memory=all_memory,
                                                                                last_memory=last_memory)
        self.repeat_explore=repeat_recommendation_mechanism
        # batch_size * 2

        timeline_mask = torch.BoolTensor(seq_item == 0).to(self.device)
        repeat_recommendation_decoder = self.repeat_recommendation_decoder.forward(all_memory=all_memory,
                                                                                   last_memory=last_memory,
                                                                                   seq_item=seq_item,
                                                                                   mask=timeline_mask)
        # batch_size * num_item

        explore_recommendation_decoder = self.explore_recommendation_decoder.forward(all_memory=all_memory,
                                                                                     last_memory=last_memory,
                                                                                     seq_item=seq_item,
                                                                                     mask=timeline_mask)
        # batch_size * num_item

        prediction = repeat_recommendation_decoder * repeat_recommendation_mechanism[:, 0].unsqueeze(
            1) + explore_recommendation_decoder * repeat_recommendation_mechanism[:, 1].unsqueeze(1)
        # batch_size * num_item

        return prediction


    def calculate_loss(self, interaction):

        seq_item= interaction[self.ITEM_SEQ]
        pos_item=interaction[self.POS_ITEM_ID]
        prediction = self.forward(seq_item)
        loss = self.loss_fct((prediction + 1e-8).log(), pos_item, ignore_index=0)
        if self.joint_train is True:
            loss+=self.repeat_explore_loss(seq_item,pos_item)

        return loss


    def repeat_explore_loss(self,seq_item,pos_item):
        """

        :param seq_item: batch_size * seq_len
        :param pos_item: batch_size
        :return:
        """
        batch_size=seq_item.size(0)
        repeat,explore=torch.zeros(batch_size).to(self.device),torch.ones(batch_size).to(self.device)
        i=0
        for x,y in zip(seq_item,pos_item):
            if y in x:
                repeat[i]=1
                explore[i]=0
            i+=1
        repeat_loss=torch.mul(repeat.unsqueeze(1),torch.log(self.repeat_explore[:,0]+1e-8)).sum()
        explore_loss=torch.mul(explore.unsqueeze(1),torch.log(self.repeat_explore[:,1]+1e-8)).sum()

        return -repeat_loss-explore_loss


    def full_sort_predict(self, interaction):

        seq_item=interaction[self.ITEM_SEQ]
        prediction = self.forward(seq_item)

        return prediction


class Repeat_Explore_Mechanism(nn.Module):

    def __init__(self, hidden_size=32, seq_len=10,dropout_prob=0.5):
        super(Repeat_Explore_Mechanism, self).__init__()
        self.dropout=nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vre = nn.Linear(hidden_size, 1, bias=False)
        self.Wre = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, all_memory, last_memory):
        """

        :param all_memory: batch_size * seq_len * hidden_size
        == Ur ==>> batch_size * seq_len * hidden_size

        :param last_memory: batch_size * hidden_size
        == Wr ==>> batch_size * hidden_size
        == unsqueeze ==>> batch_size * 1 * hidden_size
        == repeat ==>> batch_size * seq_len * hidden_size

        ***
        last_memory + all_memory == tanh == output
        output: batch_size * seq_len * hidden_size

        ***
        output == Vre ==>>
        output: batch_size * seq_len * 1
        == repeat (1,1,hidden_size) ==>> batch_size * seq_len * hidden_size
        == multiply all_memory_values ==>> batch_size * seq_len * hidden_size
        == sum dim=1 ==>> batch_size * hidden_size

        ***
        output == Wre ==>>
        outpur: batch_size * 2
        == softmax ==>> batch_size * 2
        :return:
        """
        all_memory_values = all_memory

        all_memory = self.dropout(self.Ur(all_memory))

        last_memory = self.dropout(self.Wr(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output = self.tanh(all_memory + last_memory)

        output = self.Vre(output)
        output = output.repeat(1, 1, self.hidden_size)
        output = output * all_memory_values
        output = output.sum(dim=1)

        output = self.Wre(output)

        output = nn.Softmax(dim=-1)(output)

        return output


class Repeat_Recommendation_Decoder(nn.Module):

    def __init__(self, hidden_size=32, seq_len=10, num_item=40000,dropout_prob=0.5):
        super(Repeat_Recommendation_Decoder, self).__init__()
        self.dropout=nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_item = num_item
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vr = nn.Linear(hidden_size, 1)


    def forward(self, all_memory, last_memory, seq_item, mask=None):
        """

        :param all_memory: batch_size * seq_len * hidden_size
        == Ur ==>> batch_size * seq_len * hidden_size
        :param last_memory: batch_size * hidden_size
        == Wr ==>> batch_size * hidden_size
        == unsqueeze ==> batch_size * 1 * hidden_size
        == repeat ==> batch_size * seq_len * hidden_size
        ***
        output = nn.tanh( last_memory + all_memory )
        batch_size * seq_len * hidden_size
        ***
        output == Vr ==>> batch_size * seq_len * 1 == squeeze ==> batch_size * seq_len
        == mask the padding item ==>> batch_size * seq_len
        == softmax the last dim ==>> batch_size * seq_len
        ***
        reflect to the shape of: batch_size * num_item
        == unsqueeze the dim 1: batch_size * 1 * seq_len
        map: batch_size * seq_len * num_item
        ***
        output matmul map: batch_size * 1 * num_item
        == squeeze dim 1 == batch_size * num_item

        :return:
        """
        all_memory = self.dropout(self.Ur(all_memory))

        last_memory = self.dropout(self.Wr(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output = self.tanh(last_memory + all_memory)

        output = self.Vr(output).squeeze(2)

        if mask is not None:
            output.masked_fill_(mask, -1e9)

        output = nn.Softmax(dim=-1)(output)
        output = output.unsqueeze(1)

        map = build_map(seq_item, max=self.num_item)
        output = torch.matmul(output, map).squeeze(1)
        output = output.squeeze(1)

        return output


class Explore_Recommendation_Decoder(nn.Module):


    def __init__(self, hidden_size, seq_len, num_item,dropout_prob=0.5):
        super(Explore_Recommendation_Decoder, self).__init__()
        self.dropout=nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_item = num_item
        self.We = nn.Linear(hidden_size, hidden_size)
        self.Ue = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.Ve = nn.Linear(hidden_size, 1)
        self.matrix_for_explore = nn.Linear(2 * self.hidden_size, self.num_item, bias=False)


    def forward(self, all_memory, last_memory, seq_item, mask=None):
        """

        :param all_memory: batch_size * seq_len * hidden_size
        == Ue ==>> batch_size * seq_len * hidden_size
        :param last_memory: batch_size * hidden_size
        == We ==>> batch_size * hidden_size
        == unsqueeze ==>> batch_size * 1 * hidden_size
        == repeat  ==>> batch_size * seq_len * hidden_size
        ***
        output = nn.tanh( last_memory + all_memory )
        == Ve ==>> batch_size * seq_len * 1
        == repeat ==>> batch_size * seq_len * hidden_size
        ***
        output = all_memory_values * output : batch_size * seq_len * hidden_size
        == sum dim=1 ==>> batch_size * hidden_size
        ***
        concat output and last_memory:== >> batch_size * 2_mul_hidden_size
        ***
        == matrix for explore ==>> batch_size * num_item
        :return:
        """
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.Ue(all_memory))

        last_memory = self.dropout(self.We(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)

        output = self.tanh(all_memory + last_memory)
        output = self.Ve(output).squeeze(-1)

        if mask is not None:
            output.masked_fill_(mask, -1e9)

        output = output.unsqueeze(-1)

        output = nn.Softmax(dim=1)(output)
        output = output.repeat(1, 1, self.hidden_size)
        output = (output * all_memory_values).sum(dim=1)
        output = torch.cat([output, last_memory_values], dim=1)
        output = self.dropout(self.matrix_for_explore(output))

        map = build_map(seq_item, max=self.num_item)
        explore_mask = torch.bmm((seq_item > 0).float().unsqueeze(1), map).squeeze(1)
        output = output.masked_fill(explore_mask.bool(), float('-inf'))
        output = nn.Softmax(1)(output)

        return output


def build_map(b_map, max=None):
    """
    project the b_map to the place where it in should be
    like this:
        seq_item A: [3,4,5]   n_items: 6
        after map: A
        [0,0,1,0,0,0]
        [0,0,0,1,0,0]
        [0,0,0,0,1,0]

    batch_size * seq_len ==>> batch_size * seq_len * n_item

    use in RepeatNet:
    [3,4,5] matmul [0,0,1,0,0,0]
                   [0,0,0,1,0,0] ==>>>   [0,0,3,4,5,0]  it works in the RepeatNet when project the seq item into all items
                   [0,0,0,0,1,0]
    batch_size * 1 * seq_len matmul batch_size * seq_len * n_item ==>> batch_size * 1 * n_item
    """
    batch_size, b_len = b_map.size()
    if max is None:
        max = b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    b_map_.requires_grad = False
    return b_map_