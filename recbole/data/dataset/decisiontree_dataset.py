# @Time   : 2020/12/17
# @Author : Chen Yang
# @Email  : 254170321@qq.com

"""
recbole.data.decisiontree_dataset
##########################
"""

from recbole.data.dataset import Dataset
from recbole.utils import FeatureType


class DecisionTreeDataset(Dataset):
    """:class:`DecisionTreeDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and 

    Attributes:

    """

    def __init__(self, config):
        super().__init__(config)

    def _judge_token_and_convert(self, feat):
        # get columns whose type is token
        col_list = []
        for col_name in feat:
            if col_name == self.uid_field or col_name == self.iid_field:
                continue
            if self.field2type[col_name] == FeatureType.TOKEN:
                col_list.append(col_name)
            elif self.field2type[col_name] in {FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ}:
                feat = feat.drop([col_name], axis=1, inplace=False)

        # get hash map
        for col in col_list:
            self.hash_map[col] = dict({})
            self.hash_count[col] = 0

        del_col = []
        for col in self.hash_map:
            if col in feat.keys():
                for value in feat[col]:
                    # print(value)
                    if value not in self.hash_map[col]:
                        self.hash_map[col][value] = self.hash_count[col]
                        self.hash_count[col] = self.hash_count[col] + 1
                        if self.hash_count[col] > self.config['token_num_threshold']:
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
        if self.config['convert_token_to_onehot']:
            for feat_name in ['inter_feat', 'user_feat', 'item_feat']:
                feat = getattr(self, feat_name)
                if feat is not None:
                    feat = self._judge_token_and_convert(feat)
                setattr(self, feat_name, feat)

    def _from_scratch(self):
        super()._from_scratch()
        self._convert_token_to_hash()
