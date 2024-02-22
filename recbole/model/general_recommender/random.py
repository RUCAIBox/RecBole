# -*- coding: utf-8 -*-
# @Time   : 2023/03/01
# @Author : João Felipe Guedes
# @Email  : guedes.joaofelipe@poli.ufrj.br
# UPDATE

r"""
Random
################################################

"""

import torch
import random

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class Random(GeneralRecommender):
    """Random is an fundamental model that recommends random items."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(Random, self).__init__(config, dataset)
        torch.manual_seed(config["seed"] + self.n_users + self.n_items)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        return torch.rand(len(interaction), device=self.device).squeeze(-1)

    def full_sort_predict(self, interaction):
        batch_user_num = interaction[self.USER_ID].shape[0]
        result = torch.rand(self.n_items, 1).to(torch.float64)
        result = torch.repeat_interleave(result.unsqueeze(0), batch_user_num, dim=0)
        return result.view(-1)
