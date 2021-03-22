# -*- coding: utf-8 -*-
# @Time   : 2020/1/17
# @Author : Chen Yang
# @Email  : 254170321@qq.com

r"""
recbole.model.exlib_recommender.lightgbm
##########################################
"""

import lightgbm as lgb
from recbole.utils import ModelType, InputType


class lightgbm(lgb.Booster):
    r"""lightgbm is inherited from lgb.Booster

    """
    type = ModelType.DECISIONTREE
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(lgb.Booster, self).__init__()

    def to(self, device):
        return self
