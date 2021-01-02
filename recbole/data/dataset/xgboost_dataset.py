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
        # get columns whose type is token
        col_list = []
        for col_name in feat:
            if col_name == self.uid_field or col_name == self.iid_field:
                continue
            if self.field2type[col_name] == FeatureType.TOKEN:
                col_list.append(col_name)
            elif self.field2type[col_name] == FeatureType.TOKEN_SEQ or self.field2type[col_name] == FeatureType.FLOAT_SEQ:
                feat = feat.drop([col_name], axis=1, inplace=False)

        # get hash map
        for col in col_list:
            self.hash_map[col] = dict({})
            self.hash_count[col] = 0

        del_col = []
        for col in self.hash_map:
            if col in feat.keys():
                for value in feat[col]:
                    #print(value)
                    if value not in self.hash_map[col]:
                        self.hash_map[col][value] = self.hash_count[col]
                        self.hash_count[col] = self.hash_count[col] + 1
                        if self.hash_count[col] > self.config['token_num_threhold']:
                            del_col.append(col)
                            break

        for col in del_col:
            del self.hash_count[col]
            del self.hash_map[col]
            col_list.remove(col)
        self.convert_col_list.extend(col_list)

        # transform the original data
        for col in self.hash_map.keys():
            if col in feat.keys():
                feat[col] = feat[col].map(self.hash_map[col])

        return feat

    def _convert_token_to_hash(self):
        """Convert the data of token type to hash form
        
        """
        self.hash_map = {}
        self.hash_count = {}
        self.convert_col_list = []
        if self.config['convert_token_to_onehot'] == True:
            feat_list = []
            for feat in (self.inter_feat, self.user_feat, self.item_feat):
                if feat is not None:
                    feat = self._judge_token_and_convert(feat)
                feat_list.append(feat)
            self.inter_feat = feat_list[0]
            self.user_feat = feat_list[1]
            self.item_feat = feat_list[2]

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.debug('Loading {} from scratch'.format(self.__class__))

        self._get_preset()
        self._get_field_from_config()
        self._load_data(self.dataset_name, self.dataset_path)
        self._data_processing()
        self._convert_token_to_hash()
        self._change_feat_format()

    def join(self, df):
        """Given interaction feature, join user/item feature into it.

        Args:
            df (Interaction): Interaction feature to be joint.

        Returns:
            Interaction: Interaction feature after joining operation.
        """
        if self.user_feat is not None and self.uid_field in df:
            df.update(self.user_feat[df[self.uid_field]])
        if self.item_feat is not None and self.iid_field in df:
            df.update(self.item_feat[df[self.iid_field]])
        return df

    def __getitem__(self, index, join=True):
        df = self.inter_feat[index]
        return self.join(df) if join else df
