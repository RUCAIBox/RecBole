# @Time   : 2020/12/17
# @Author : Chen Yang
# @Email  : 254170321@qq.com

"""
recbole.data.xgboost_dataset
##########################
"""

import numpy as np
import pandas as pd

from recbole.data.dataset import Dataset
from recbole.utils import FeatureType
from recbole.data.interaction import Interaction
from recbole.data.utils import dlapi


class XgboostDataset(Dataset):
    """:class:`XgboostDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and 

    Attributes:

    """
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)

    def _judge_token_and_convert(self, feat):
        col_list = []
        for col_name in feat:
            if col_name == self.uid_field or col_name == self.iid_field:
                continue
            if self.field2type[col_name] == FeatureType.TOKEN:
                col_list.append(col_name)
            elif self.field2type[col_name] == FeatureType.TOKEN_SEQ or self.field2type[col_name] == FeatureType.FLOAT_SEQ:
                feat = feat.drop([col_name], axis=1, inplace=False)
        feat = pd.get_dummies(feat, sparse = True, columns = col_list)
        for col_name in feat.columns.values.tolist():
            if col_name not in self.field2type.keys():
                self.field2type[col_name] = FeatureType.TOKEN
        return feat

    def _convert_token_to_onehot(self):
        """Convert the data of token type to onehot form
        
        """
        if self.config['convert_token_to_onehot'] == True:
            feat_list = []
            for feat in (self.inter_feat, self.user_feat, self.item_feat):
                feat = self._judge_token_and_convert(feat)
                feat_list.append(feat)
            self.inter_feat_xgb = feat_list[0]
            self.user_feat_xgb = feat_list[1]
            self.item_feat_xgb = feat_list[2]
            self.inter_feat = self.inter_feat_xgb
            self.user_feat = self.user_feat_xgb
            self.item_feat = self.item_feat
        else:
            self.inter_feat_xgb = self.inter_feat
            self.user_feat_xgb = self.user_feat
            self.item_feat_xgb = self.item_feat

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.debug('Loading {} from scratch'.format(self.__class__))

        self._get_preset()
        self._get_field_from_config()
        self._load_data(self.dataset_name, self.dataset_path)
        self._data_processing()
        self._convert_token_to_onehot()
        self._change_feat_format()

    def join(self, df):
        """Given interaction feature, join user/item feature into it.

        Args:
            df (pandas.DataFrame): Interaction feature to be joint.

        Returns:
            pandas.DataFrame: Interaction feature after joining operation.
        """
        if self.user_feat is not None and self.uid_field in df:
            df = pd.merge(df, self.user_feat_xgb, on=self.uid_field, how='left', suffixes=('_inter', '_user'))
        if self.item_feat is not None and self.iid_field in df:
            df = pd.merge(df, self.item_feat_xgb, on=self.iid_field, how='left', suffixes=('_inter', '_item'))
        return df

    def __getitem__(self, index, join=True):
        df = self.inter_feat_xgb[index]
        return self.join(df) if join else df
