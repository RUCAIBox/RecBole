# -*- coding: utf-8 -*-
# @Time   : 2020/6/25 15:47
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : abstract_recommender.py

import numpy as np
import torch.nn as nn


class AbstractRecommender(nn.Module):
    """
    Base class for all models
    """
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def train_model(self, *inputs):
        """
         Calculate Train loss

        :return: Model train loss
        """
        raise NotImplementedError

    def predict(self, *inputs):
        """
         Result prediction for testing and evaluating

        :return: Model predict
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
