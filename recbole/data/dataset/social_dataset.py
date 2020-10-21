# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/9, 2020/9/15, 2020/9/22
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.social_dataset
##########################
"""

import os

import numpy as np
from scipy.sparse import coo_matrix

from recbole.data.dataset import Dataset
from recbole.data.utils import dlapi
from recbole.utils import FeatureSource


class SocialDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)

    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.source_field = self.config['SOURCE_ID_FIELD']
        self.target_field = self.config['TARGET_ID_FIELD']
        self._check_field('source_field', 'target_field')

        self.logger.debug('source_id_field: {}'.format(self.source_field))
        self.logger.debug('target_id_field: {}'.format(self.target_field))

    def _load_data(self, token, dataset_path):
        super()._load_data(token, dataset_path)
        self.net_feat = self._load_net(self.dataset_name, self.dataset_path)

    def _build_feat_list(self):
        feat_list = super()._build_feat_list()
        if self.net_feat is not None:
            feat_list.append(self.net_feat)
        return feat_list

    def _load_net(self, dataset_name, dataset_path): 
        net_file_path = os.path.join(dataset_path, '{}.{}'.format(dataset_name, 'net'))
        if os.path.isfile(net_file_path):
            net_feat = self._load_feat(net_file_path, FeatureSource.NET)
            if net_feat is None:
                raise ValueError('.net file exist, but net_feat is None, please check your load_col')
            return net_feat
        else:
            raise ValueError('File {} not exist'.format(net_file_path))
            
    def _get_fields_in_same_space(self):
        fields_in_same_space = super()._get_fields_in_same_space()
        fields_in_same_space = [_ for _ in fields_in_same_space if (self.source_field not in _) and
                                                                   (self.target_field not in _)]
        for field_set in fields_in_same_space:
            if self.uid_field in field_set:
                field_set.update({self.source_field, self.target_field})

        return fields_in_same_space

    @dlapi.set()
    def net_graph(self, form='coo', value_field=None):
        args = [self.net_feat, self.source_field, self.target_field, form, value_field]
        if form in ['coo', 'csr']:
            return self._create_sparse_matrix(*args)
        elif form in ['dgl', 'pyg']:
            return self._create_graph(*args)
        else:
            raise NotImplementedError('net graph format [{}] has not been implemented.')

    def __str__(self):
        info = [super().__str__(),
                'The number of connections of social network: {}'.format(len(self.net_feat))]
        return '\n'.join(info)
