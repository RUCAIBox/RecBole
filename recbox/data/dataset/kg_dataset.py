# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

import os

import numpy as np
import pandas as pd

from .dataset import Dataset
from ...utils import FeatureSource, FeatureType


class KnowledgeBasedDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)

    def _from_scratch(self, config):
        self.logger.debug('Loading dataset from scratch')

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

        self.logger.debug('uid_field: {}'.format(self.uid_field))
        self.logger.debug('iid_field: {}'.format(self.iid_field))

        self.head_entity_field = self.config['HEAD_ENTITY_ID_FIELD']
        self.tail_entity_field = self.config['TAIL_ENTITY_ID_FIELD']
        self.relation_field = self.config['RELATION_ID_FIELD']
        self.entity_field = self.config['ENTITY_ID_FIELD']
        self._check_field('head_entity_field', 'tail_entity_field', 'relation_field', 'entity_field')

        self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.dataset_name, self.dataset_path)
        self.feat_list = [feat for feat in [self.inter_feat, self.user_feat, self.item_feat] if feat is not None]

        self.kg_feat = self._load_kg(self.dataset_name, self.dataset_path)
        self.item2entity, self.entity2item = self._load_link(self.dataset_name, self.dataset_path)

        self._filter_by_inter_num()
        self._filter_by_field_value()
        self._reset_index()
        self._remap_ID_all()
        self._user_item_feat_preparation()

        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()

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
            entity_id_token[i] = self.item2entity[id2item[entity_id_token[i]]]

        for ent_field in [self.head_entity_field, self.tail_entity_field, self.entity_field]:
            self.field2id_token[ent_field] = entity_id_token

        self.field2source[self.entity_field] = FeatureSource.KG
        self.field2type[self.entity_field] = FeatureType.TOKEN
        self.field2seqlen[self.entity_field] = 1

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
