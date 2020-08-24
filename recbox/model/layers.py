# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 16:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : layers.py

# UPDATE:
# @Time   : 2020/8/24 14:58
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

"""
Common Layers in recommender system
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import xavier_normal_


class MLPLayers(nn.Module):
    """ MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'
                      candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:
        - Input: (N, *, H_{in}) where * means any number of additional dimensions
          H_{in} must equal to the first value in `layers`
        - Output: (N, *, H_{out}) where H_{out} equals to the last value in `layers`

    Examples::

        >> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >> input = torch.randn(128, 64)
        >> output = m(input)
        >> print(output.size())
        >> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout=0, activation='relu', bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            if self.activation.lower() == 'sigmoid':
                mlp_modules.append(nn.Sigmoid())
            elif self.activation.lower() == 'tanh':
                mlp_modules.append(nn.Tanh())
            elif self.activation.lower() == 'relu':
                mlp_modules.append(nn.ReLU())
            elif self.activation.lower() == 'leekyrelu':
                mlp_modules.append(nn.LeakyReLU())
            elif self.activation.lower() == 'none':
                pass
            else:
                warnings.warn('Received unrecognized activation function, set default activation function', UserWarning)


        self.mlp_layers = nn.Sequential(*mlp_modules)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class FMEmbedding(nn.Module):
    """
        Input shape
        - A 3D tensor with shape:``(batch_size,field_size)``.

        Output shape
        - 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FMEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class BaseFactorizationMachine(nn.Module):
    """
        Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

        Output shape
        - 3D tensor with shape: ``(batch_size,1)`` or (batch_size, embed_dim).
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x ** 2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class BiGNNLayer(nn.Module):

    def __init__(self, in_dim, out_dim):

        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        L1 = lap_matrix + eye_matrix

        inter_part1 = self.linear(torch.sparse.mm(L1, features))

        inter_feature = torch.sparse.mm(lap_matrix, features)
        inter_feature = torch.mul(inter_feature, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2


class AttLayer(nn.Module):
    """
        Input shape
        - A 3D tensor with shape:``(batch_size, M, embed_dim)``.

        Output shape
        - 3D tensor with shape: ``(batch_size, M)`` .
    """

    def __init__(self, in_dim, att_dim):

        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_singal = self.w(infeatures)  # [batch_size, M, att_dim]
        att_singal = fn.relu(att_singal)  # [batch_size, M, att_dim]

        att_singal = torch.mul(att_singal, self.h)  # [batch_size, M, att_dim]
        att_singal = torch.sum(att_singal, dim=2)  # [batch_size, M]
        att_singal = fn.softmax(att_singal, dim=1)  # [batch_size, M]

        return att_singal

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_head, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_head, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_head, bias=False)
        self.fc = nn.Linear(self.n_head * self.d_v, self.d_model, bias=False)


    def scale_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask, -1e9) # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


    def forward(self, input_Q, input_K, input_V, mask = None):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        context, attn = self.scale_dot_product_attention(Q, K, V, mask)
        context = context.transpose(1,2).reshape(batch_size, -1, self.n_head * self.d_v)
        output = self.fc(context)
        return nn.LayerNorm(self.d_model)(output + residual), attn


