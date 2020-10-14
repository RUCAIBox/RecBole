# -*- coding: utf-8 -*-
# @Time   : 2020/10/2
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn


r"""
recbox.model.sequential_recommender.nextitnet
################################################

Reference:
Fajie Yuan et al., "A Simple Convolutional Generative Network for Next Item Recommendation" in WSDM 2019.

Reference code:
https://github.com/fajieyuan/nextitnet
https://github.com/initlisk/nextitnet_pytorch

"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_

from recbox.utils import InputType
from recbox.model.loss import RegLoss
from recbox.model.abstract_recommender import SequentialRecommender


class NextItNet(SequentialRecommender):
    r"""The network architecture of the NextItNet model is formed of a stack of holed convolutional layers, which can
    efficiently increase the receptive fields without relying on the pooling operation.
    Also residual block structure is used to ease the optimization for much deeper networks.

    Note:
        As paper said, for comparison purpose, we only predict the next one item in our evaluation,
        and then stop the generating process. Although the number of parameters in residual block (a) is less
        than it in residual block (b), the performance of b is better than a.
        So in our model, we use residual block (b).
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NextItNet, self).__init__()

        # load parameters info
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.TARGET_ITEM_ID =  self.ITEM_ID
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.embedding_size = config['embedding_size']
        self.residual_channels = config['embedding_size']
        self.block_num = config['block_num']
        self.dilations = config['dilations'] * self.block_num
        self.kernel_size = config['kernel_size']
        self.onecall = config['onecall']
        self.reg_weight = config['reg_weight']
        self.item_count = dataset.item_num

        # define loss function
        self.criterion = nn.CrossEntropyLoss()
        self.reg_loss = RegLoss()

        # item embeddings
        self.item_list_embedding = nn.Embedding(self.item_count, self.embedding_size)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [ResidualBlock_b(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size,
                            dilation=dilation) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.embedding_size)

        # weight initialization
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.item_count)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, interaction):
        # Embedding Look-up
        inputs = self.item_list_embedding(interaction[self.ITEM_ID_LIST]) # [batch_size, seq_len, embed_size]

        # Residual locks
        dilate_outputs = self.residual_blocks(inputs)

        if self.onecall:   # Extract the last item
            hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels)  # [batch_size, embed_size]
        else:
            hidden = dilate_outputs.view(-1, self.residual_channels)  # [batch_size*seq_len, embed_size]

        pred_item_emb = self.final_layer(hidden)   # [batch_size, embedding_size]

        return pred_item_emb

    def get_item_lookup_table(self):
        r"""Get the transpose of item_list_embedding.weight，size: (embedding_size * item_count)
        Used to calculate the score for each item with the predict_behavior_emb
        """
        return self.item_list_embedding.weight.t()

    def reg_loss_rb(self):
        r"""
        L2 loss on residual blocks
        """
        loss_rb = 0
        if self.reg_weight > 0.0:
            for name, parm in self.residual_blocks.named_parameters():
                if name.endswith('weight'):
                    loss_rb += torch.norm(parm,2)
        return self.reg_weight * loss_rb

    def calculate_loss(self, interaction):
        target_id = interaction[self.TARGET_ITEM_ID]
        pred = self.forward(interaction)
        logits = torch.matmul(pred, self.get_item_lookup_table())
        loss = self.criterion(logits, target_id)
        reg_loss = self.reg_loss([self.item_list_embedding.weight,self.final_layer.weight])
        loss = loss + self.reg_weight * reg_loss + self.reg_loss_rb()
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        pred = self.forward(interaction)
        scores = torch.matmul(pred, self.get_item_lookup_table())
        return scores


class ResidualBlock_a(nn.Module):
    r"""
    Residual block (a) in the paper
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_a, self).__init__()

        half_channel = out_channel//2
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv1 = nn.Conv2d(in_channel, half_channel, kernel_size=(1, 1), padding=0)


        self.ln2 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(half_channel, half_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)

        self.ln3 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv3 = nn.Conv2d(half_channel, out_channel, kernel_size=(1, 1), padding=0)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]

        out = F.relu(self.ln1(x))
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.conv1(out).squeeze(2).permute(0, 2, 1)

        out2 = F.relu(self.ln2(out))
        out2 = self.conv_pad(out2, self.dilation)
        out2 = self.conv2(out2).squeeze(2).permute(0, 2, 1)

        out3 = F.relu(self.ln3(out2))
        out3 = out3.permute(0, 2, 1).unsqueeze(2)
        out3 = self.conv3(out3).squeeze(2).permute(0, 2, 1)
        return out3 + x

    def conv_pad(self, x, dilation):  # x: [batch_size, seq_len, embed_size]
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this funtion.
        """
        inputs_pad = x.permute(0, 2, 1)       # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))  # padding opration  args：(left,right,top,bottom)
        inputs_pad = pad(inputs_pad)          # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad


class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation*2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)   # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)  # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation*2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad
