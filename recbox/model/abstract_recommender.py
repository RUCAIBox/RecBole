# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/6, 2020/8/25
# @Author : Shanlei Mu, Yupeng Hou
# @Email  : slmu@ruc.edu.cn, houyupeng@ruc.edu.cn

import numpy as np
import torch
import torch.nn as nn
from ..utils import ModelType


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
    type = ModelType.GENERAL
    def __init__(self):
        super(GeneralRecommender, self).__init__()


class SequentialRecommender(AbstractRecommender):
    type = ModelType.SEQUENTIAL
    def __init__(self):
        super(SequentialRecommender, self).__init__()


class KnowledgeRecommender(AbstractRecommender):
    type = ModelType.KNOWLEDGE
    def __init__(self, config, dataset):
        super(KnowledgeRecommender, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        # todo: 和data部分对接
        self.ENTITY_ID = config['ENTITY_ID_FIELD']
        self.RELATION_ID = config['RELATION_ID_FIELD']
        self.HEAD_ENTITY_ID = config['HEAD_ENTITY_ID_FIELD']
        self.TAIL_ENTITY_ID = config['TAIL_ENTITY_ID_FIELD']
        self.NEG_TAIL_ENTITY_ID = config['NEG_PREFIX'] + self.TAIL_ENTITY_ID

        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID)
