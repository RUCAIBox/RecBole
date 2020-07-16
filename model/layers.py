# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 16:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : layers.py

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

    def __init__(self, layers, dropout=0, activation='none'):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

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
                warnings.warn('Received unrecognized activation function, set default activation function'
                              , UserWarning)

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
        super(FMEmbedding).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.embedding.weight)

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class FMFirstOrderLinear(nn.Module):
    """
        Input shape
        - A 3D tensor with shape:``(batch_size,field_size)``.

        Output shape
        - 3D tensor with shape: ``(batch_size,output_dim)``.
    """
    def __init__(self, field_dims, offsets, output_dim=1):

        super(FMFirstOrderLinear).__init__()

        self.w = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim, )))
        self.offsets = offsets

        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.w.weight)

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = torch.sum(self.w(input_x), dim=1) + self.bias
        return output


class BaseFactorizationMachine(nn.Module):
    """
        Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

        Output shape
        - 3D tensor with shape: ``(batch_size,1)`` or (batch_size, embed_dim).
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x ** 2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


