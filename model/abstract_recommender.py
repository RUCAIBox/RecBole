# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

import numpy as np
import torch
import torch.nn as nn
from utils import ModelType


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

    def calculate_loss(self, *inputs):
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


class GeneralRecommender(AbstractRecommender):
    def __init__(self):
        super(GeneralRecommender, self).__init__()
        self.type = ModelType.GENERAL


class SequentialRecommender(AbstractRecommender):
    def __init__(self):
        super(SequentialRecommender, self).__init__()
        self.type = ModelType.SEQUENTIAL


class KnowledgeRecommender(AbstractRecommender):
    def __init__(self):
        super(KnowledgeRecommender, self).__init__()
        self.type = ModelType.KNOWLEDGE
