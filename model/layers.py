# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 16:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : layers.py

"""
Common Layers in recommender system
"""

import warnings
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

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)
