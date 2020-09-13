# -*- coding: utf-8 -*-
# @Time    : 2020/9/11
# @Author  : Shanlei Mu
# @Email   : msl@ruc.edu.cn

import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
