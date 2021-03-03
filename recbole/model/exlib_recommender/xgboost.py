# -*- coding: utf-8 -*-
# @Time   : 2020/11/19
# @Author : Chen Yang
# @Email  : 254170321@qq.com

r"""
recbole.model.exlib_recommender.xgboost
########################################
"""

import xgboost as xgb
from recbole.utils import ModelType, InputType


class xgboost(xgb.Booster):
    r"""xgboost is inherited from xgb.Booster

    """
    type = ModelType.DECISIONTREE
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(params=None, cache=(), model_file=None)

    def to(self, device):
        return self
