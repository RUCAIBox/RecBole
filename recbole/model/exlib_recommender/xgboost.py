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


class XGBoost(xgb.Booster):
    r"""XGBoost is inherited from xgb.Booster"""

    type = ModelType.DECISIONTREE
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(params=None, cache=(), model_file=None)

    def to(self, device):
        return self

    def load_state_dict(self, model_file):
        r"""Load state dictionary

        Args:
            model_file (str): file path of saved model

        """
        self.load_model(model_file)

    def load_other_parameter(self, other_parameter):
        r"""Load other parameters"""
        pass
