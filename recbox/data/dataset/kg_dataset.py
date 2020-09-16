# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16, 2020/9/15
# @Author : Yupeng Hou, Xingyu Pan
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn

import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch

from recbox.data.dataset.dataset import Dataset
from recbox.utils import FeatureSource, FeatureType


class KnowledgeBasedDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)

    def _from_scratch(self, config):
        self.logger.debug('Loading kg dataset from scratch')

        self.dataset_path = config['data_path']
        self._fill_nan_flag = self.config['fill_nan']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2seqlen = config['seq_len'] or {}

        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        self.logger.debug('uid_field: {}'.format(self.uid_field))
        self.logger.debug('iid_field: {}'.format(self.iid_field))

        self.head_entity_field = self.config['HEAD_ENTITY_ID_FIELD']
        self.tail_entity_field = self.config['TAIL_ENTITY_ID_FIELD']
        self.relation_field = self.config['RELATION_ID_FIELD']
        self.entity_field = self.config['ENTITY_ID_FIELD']
        self._check_field('head_entity_field', 'tail_entity_field', 'relation_field', 'entity_field')

        self.logger.debug('relation_field: {}'.format(self.relation_field))
        self.logger.debug('entity_field: {}'.format(self.entity_field))

        self._preloaded_weight = {}

        self.benchmark_filename_list = config['benchmark_filename']
        if self.benchmark_filename_list is None:
            self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.dataset_name, self.dataset_path)
        else:
            self.inter_feat, self.user_feat, self.item_feat, self.file_size_list = self._load_benchmark_file(self.dataset_name, self.dataset_path, self.benchmark_filename_list)

        self.kg_feat = self._load_kg(self.dataset_name, self.dataset_path)
        self.item2entity, self.entity2item = self._load_link(self.dataset_name, self.dataset_path)
        self.feat_list = self._build_feat_list()

        if self.benchmark_filename_list is None:
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
        return [feat for feat in [self.inter_feat, self.user_feat, self.item_feat, self.kg_feat] if feat is not None]

    def _restore_saved_dataset(self, saved_dataset):
        raise NotImplementedError()

    def save(self, filepath):
        raise NotImplementedError()

    def _load_kg(self, token, dataset_path):
        self.logger.debug('loading kg from [{}]'.format(dataset_path))
        kg_path = os.path.join(dataset_path, '{}.{}'.format(token, 'kg'))
        if not os.path.isfile(kg_path):
            raise ValueError('[{}.{}] not found in [{}]'.format(token, 'kg', dataset_path))
        df = pd.read_csv(kg_path, delimiter=self.config['field_separator'])
        field_names = []
        for field_type in df.columns:
            field, ftype = field_type.split(':')
            field_names.append(field)
            assert ftype == 'token', 'kg data requires fields with type token'
            self.field2source[field] = FeatureSource.KG
            self.field2type[field] = FeatureType.TOKEN
            self.field2seqlen[field] = 1
        df.columns = field_names
        self._check_kg(df)
        return df

    def _check_kg(self, kg):
        kg_warn_message = 'kg data requires field [{}]'
        assert self.head_entity_field in kg, kg_warn_message.format(self.head_entity_field)
        assert self.tail_entity_field in kg, kg_warn_message.format(self.tail_entity_field)
        assert self.relation_field in kg, kg_warn_message.format(self.relation_field)

    def _load_link(self, token, dataset_path):
        self.logger.debug('loading link from [{}]'.format(dataset_path))
        link_path = os.path.join(dataset_path, '{}.{}'.format(token, 'link'))
        if not os.path.isfile(link_path):
            raise ValueError('[{}.{}] not found in [{}]'.format(token, 'link', dataset_path))
        df = pd.read_csv(link_path, delimiter=self.config['field_separator'])
        field_names = []
        for field_type in df.columns:
            field, ftype = field_type.split(':')
            field_names.append(field)
            assert ftype == 'token', 'kg data requires fields with type token'
        df.columns = field_names
        self._check_link(df)

        item2entity, entity2item = {}, {}
        for item_id, entity_id in zip(df[self.iid_field].values, df[self.entity_field].values):
            item2entity[item_id] = entity_id
            entity2item[entity_id] = item_id
        return item2entity, entity2item

    def _check_link(self, link):
        link_warn_message = 'link data requires field [{}]'
        assert self.entity_field in link, link_warn_message.format(self.entity_field)
        assert self.iid_field in link, link_warn_message.format(self.iid_field)

    def _get_fields_in_same_space(self):
        fields_in_same_space = super()._get_fields_in_same_space()
        fields_in_same_space = [
            _ for _ in fields_in_same_space
            if (self.head_entity_field not in _) and
               (self.tail_entity_field not in _) and
               (self.entity_field not in _)
        ]
        return fields_in_same_space

    def _remap_ID_all(self):
        super()._remap_ID_all()

        item2id = {}
        for i, item_id in enumerate(self.field2id_token[self.iid_field]):
            item2id[item_id] = i

        for ent_field in [self.head_entity_field, self.tail_entity_field]:
            entity_list = self.kg_feat[ent_field].values
            entity_list = [item2id[self.entity2item[_]] if (_ in self.entity2item) and
                                                           (self.entity2item[_] in item2id)
                                                        else _ for _ in entity_list]
            self.kg_feat[ent_field] = entity_list

        fields_in_same_space = self._get_fields_in_same_space()
        self.logger.debug('fields_in_same_space: {}'.format(fields_in_same_space))

        for field_set in fields_in_same_space:
            if self.iid_field not in field_set:
                continue
            remap_list = []
            field_set.remove(self.iid_field)
            remap_list.append((self.inter_feat, self.iid_field, FeatureType.TOKEN))
            if self.item_feat is not None:
                remap_list.append((self.item_feat, self.iid_field, FeatureType.TOKEN))
            field_set.update({self.head_entity_field, self.tail_entity_field})
            for field in field_set:
                source = self.field2source[field]
                feat = getattr(self, '{}_feat'.format(source.value))
                ftype = self.field2type[field]
                remap_list.append((feat, field, ftype))
            self._remap(remap_list, overwrite=False)

        entity_id_token = self.field2id_token[self.head_entity_field]
        id2item = self.field2id_token[self.iid_field]
        for i in range(1, self.item_num):
            tmp_item_id = id2item[entity_id_token[i]]
            if tmp_item_id in self.item2entity:
                entity_id_token[i] = self.item2entity[tmp_item_id]

        for ent_field in [self.head_entity_field, self.tail_entity_field, self.entity_field]:
            self.field2id_token[ent_field] = entity_id_token

        self.field2source[self.entity_field] = FeatureSource.KG
        self.field2type[self.entity_field] = FeatureType.TOKEN
        self.field2seqlen[self.entity_field] = 1

        self.field2id_token[self.relation_field].append('[UI-Relation]')

    @property
    def relation_num(self):
        return self.num(self.relation_field)

    @property
    def entity_num(self):
        return self.num(self.entity_field)

    @property
    def head_entities(self):
        return self.kg_feat[self.head_entity_field].values

    @property
    def entities_list(self):
        return np.arange(self.entity_num)

    def _create_dgl_kg_graph(self):
        import dgl
        kg_tensor = self._dataframe_to_interaction(self.kg_feat)
        head_entity = kg_tensor[self.head_entity_field]
        tail_entity = kg_tensor[self.tail_entity_field]
        ret = dgl.graph((head_entity, tail_entity))
        for k in kg_tensor:
            if k not in [self.head_entity_field, self.tail_entity_field]:
                ret.edata[k] = kg_tensor[k]
        return ret

    def kg_graph(self, form='coo', value_field=None):
        hids = self.kg_feat[self.head_entity_field].values
        tids = self.kg_feat[self.tail_entity_field].values
        if form in ['coo', 'csr']:
            if value_field is None:
                data = np.ones(len(self.kg_feat))
            else:
                if value_field not in self.field2source:
                    raise ValueError('value_field [{}] not exist.'.format(value_field))
                if self.field2source[value_field] != FeatureSource.KG:
                    raise ValueError('value_field [{}] can only be one of the kg features'.format(value_field))
                data = self.kg_feat[value_field].values
            mat = coo_matrix((data, (hids, tids)), shape=(self.entity_num, self.entity_num))
            if form == 'coo':
                return mat
            elif form == 'csr':
                return mat.tocsr()
        elif form == 'dgl':
            return self._create_dgl_kg_graph()
        else:
            raise NotImplementedError('net matrix format [{}] has not been implemented.')

    def _create_dgl_ckg_graph(self):
        import dgl

        kg_tensor = self._dataframe_to_interaction(self.kg_feat)
        inter_tensor = self._dataframe_to_interaction(self.inter_feat)

        head_entity = kg_tensor[self.head_entity_field]
        tail_entity = kg_tensor[self.tail_entity_field]

        user = inter_tensor[self.uid_field]
        item = inter_tensor[self.iid_field]

        source = torch.cat([user, item, head_entity])
        target = torch.cat([item, user, tail_entity])

        ret = dgl.graph((source, target))

        ui_rel_num = user.shape[0]
        ui_rel_id = self.relation_num - 1
        assert self.field2id_token[self.relation_field][ui_rel_id] == '[UI-Relation]'

        kg_rel = kg_tensor[self.relation_field]
        ui_rel = torch.full((2 * ui_rel_num,), ui_rel_id, dtype=kg_rel.dtype)
        edge = torch.cat([ui_rel, kg_rel])

        ret.edata[self.relation_field] = edge
        return ret

    def ckg_graph(self, form='coo', value_field=None):
        hids = self.kg_feat[self.head_entity_field].values
        tids = self.kg_feat[self.tail_entity_field].values

        uids = self.inter_feat[self.uid_field].values
        iids = self.inter_feat[self.iid_field].values

        ui_rel_num = len(uids)
        ui_rel_id = self.relation_num - 1
        assert self.field2id_token[self.relation_field][ui_rel_id] == '[UI-Relation]'

        source = np.concatenate([uids, iids, hids])
        target = np.concatenate([iids, uids, tids])

        if form in ['coo', 'csr']:
            if value_field is None:
                data = np.ones(len(source))
            else:
                if value_field != self.relation_field:
                    raise ValueError('v alue_field [{}] can only be [{}] in ckg_graph.'.format(value_field, self.relation_field))

                kg_rel = self.kg_feat[value_field].values
                ui_rel = np.linspace(ui_rel_id, ui_rel_id, 2*ui_rel_num, dtype=kg_rel.dtype)
                data = np.concatenate([ui_rel, kg_rel])
            mat = coo_matrix((data, (source, target)), shape=(self.entity_num, self.entity_num))
            if form == 'coo':
                return mat
            elif form == 'csr':
                return mat.tocsr()
        elif form == 'dgl':
            return self._create_dgl_ckg_graph()
        else:
            raise NotImplementedError('net matrix format [{}] has not been implemented.')
