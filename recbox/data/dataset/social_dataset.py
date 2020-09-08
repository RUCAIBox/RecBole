# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/8, 2020/9/3
# @Author : Yupeng Hou, Xingyu Pan
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn

import os
from logging import getLogger

import numpy as np
from scipy.sparse import coo_matrix

from .dataset import Dataset
from ...utils import FeatureSource


class SocialDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()

        if saved_dataset is None:
            self._from_scratch(config)
        else:
            self._restore_saved_dataset(saved_dataset)

    def _from_scratch(self, config):
        self.logger.debug('Loading social dataset from scratch')

        self.dataset_path = config['data_path']
        self._fill_nan_flag = self.config['fill_nan']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2seqlen = config['seq_len'] or {}

        self.model_type = self.config['MODEL_TYPE']
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        self.source_field = self.config['SOURCE_ID_FIELD']
        self.target_field = self.config['TARGET_ID_FIELD']
        self._check_field('source_field', 'target_field')

        self.logger.debug('uid_field: {}'.format(self.uid_field))
        self.logger.debug('iid_field: {}'.format(self.iid_field))
        self.logger.debug('source_id_field: {}'.format(self.source_field))
        self.logger.debug('target_id_field: {}'.format(self.target_field))

        self._preloaded_weight = {}

        self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.dataset_name, self.dataset_path)
        self.net_feat = self._load_net(self.dataset_name, self.dataset_path)
        self.feat_list = self._build_feat_list()

        self._filter_by_inter_num()
        self._filter_by_field_value()
        self._reset_index()
        self._remap_ID_all()
        self._user_item_feat_preparation()
        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()
        self._preload_weight_matrix()

    def _build_feat_list(self):
        return [feat for feat in [self.inter_feat, self.user_feat, self.item_feat, self.net_feat] if feat is not None]

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

    def _create_dgl_graph(self):
        import dgl
        net_tensor = self._dataframe_to_interaction(self.net_feat)
        source = net_tensor[self.source_field]
        target = net_tensor[self.target_field]
        ret = dgl.graph((source, target))
        for k in net_tensor:
            if k not in [self.source_field, self.target_field]:
                ret.edata[k] = net_tensor[k]
        return ret

    def net_matrix(self, form='coo', value_field=None):
        sids = self.net_feat[self.source_field].values
        tids = self.net_feat[self.target_field].values
        if form in ['coo', 'csr']:
            if value_field is None:
                data = np.ones(len(self.net_feat))
            else:
                if value_field not in self.field2source:
                    raise ValueError('value_field [{}] not exist.'.format(value_field))
                if self.field2source[value_field] != FeatureSource.NET:
                    raise ValueError('value_field [{}] can only be one of the net features'.format(value_field))
                data = self.net_feat[value_field].values
            mat = coo_matrix((data, (sids, tids)), shape=(self.user_num, self.user_num))
            if form == 'coo':
                return mat
            elif form == 'csr':
                return mat.tocsr()
        elif form == 'dgl':
            return self._create_dgl_graph()
        else:
            raise NotImplementedError('net matrix format [{}] has not been implemented.')

    def __str__(self):
        info = []
        if self.uid_field:
            info.extend(['The number of users: {}'.format(self.user_num),
                         'Average actions of users: {}'.format(self.avg_actions_of_users)])
        if self.iid_field:
            info.extend(['The number of items: {}'.format(self.item_num),
                         'Average actions of items: {}'.format(self.avg_actions_of_items)])
        info.append('The number of connections of social network: {}'.format(len(self.net_feat)))
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            info.append('The sparsity of the dataset: {}%'.format(self.sparsity * 100))

        info.append('Remain Fields: {}'.format(list(self.field2type)))
        return '\n'.join(info)
