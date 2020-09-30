# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/23, 2020/9/15, 2020/9/22
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn

import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch

from recbox.data.dataset import Dataset
from recbox.utils import FeatureSource, FeatureType


class KnowledgeBasedDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)

    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.head_entity_field = self.config['HEAD_ENTITY_ID_FIELD']
        self.tail_entity_field = self.config['TAIL_ENTITY_ID_FIELD']
        self.relation_field = self.config['RELATION_ID_FIELD']
        self.entity_field = self.config['ENTITY_ID_FIELD']
        self._check_field('head_entity_field', 'tail_entity_field', 'relation_field', 'entity_field')

        self.logger.debug('relation_field: {}'.format(self.relation_field))
        self.logger.debug('entity_field: {}'.format(self.entity_field))

    def _load_data(self, token, dataset_path):
        super()._load_data(token, dataset_path)
        self.kg_feat = self._load_kg(self.dataset_name, self.dataset_path)
        self.item2entity, self.entity2item = self._load_link(self.dataset_name, self.dataset_path)

    def __str__(self):
        info = [super().__str__(),
                'The number of entities: {}'.format(self.entity_num),
                'The number of relations: {}'.format(self.relation_num),
                'The number of triples: {}'.format(len(self.kg_feat)),
                'The number of items that have been linked to KG: {}'.format(len(self.item2entity))]
        return '\n'.join(info)

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
            self.set_field_property(field, FeatureType.TOKEN, FeatureSource.KG, 1)
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
            entity_list = [item2id[self.entity2item[_]]
                           if (_ in self.entity2item) and (self.entity2item[_] in item2id)
                           else _
                           for _ in entity_list]
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

        self.set_field_property(self.entity_field, FeatureType.TOKEN, FeatureSource.KG, 1)
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
    def tail_entities(self):
        return self.kg_feat[self.tail_entity_field].values

    @property
    def relations(self):
        return self.kg_feat[self.relation_field].values

    @property
    def entities(self):
        return np.arange(self.entity_num)

    def kg_graph(self, form='coo', value_field=None):
        args = [self.kg_feat, self.head_entity_field, self.tail_entity_field, form, value_field]
        if form in ['coo', 'csr']:
            return self._create_sparse_matrix(*args)
        elif form in ['dgl', 'pyg']:
            return self._create_graph(*args)
        else:
            raise NotImplementedError('kg graph format [{}] has not been implemented.')

    def _create_ckg_sparse_matrix(self, form='coo', show_relation=False):
        user_num = self.user_num

        hids = self.kg_feat[self.head_entity_field].values + user_num
        tids = self.kg_feat[self.tail_entity_field].values + user_num

        uids = self.inter_feat[self.uid_field].values
        iids = self.inter_feat[self.iid_field].values + user_num

        ui_rel_num = len(uids)
        ui_rel_id = self.relation_num - 1
        assert self.field2id_token[self.relation_field][ui_rel_id] == '[UI-Relation]'

        src = np.concatenate([uids, iids, hids])
        tgt = np.concatenate([iids, uids, tids])

        if not show_relation:
            data = np.ones(len(src))
        else:
            kg_rel = self.kg_feat[self.relation_field].values
            ui_rel = np.full(2 * ui_rel_num, ui_rel_id, dtype=kg_rel.dtype)
            data = np.concatenate([ui_rel, kg_rel])
        node_num = self.entity_num + self.user_num
        mat = coo_matrix((data, (src, tgt)), shape=(node_num, node_num))
        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    def _create_ckg_graph(self, form='dgl', show_relation=False):
        user_num = self.user_num

        kg_tensor = self._dataframe_to_interaction(self.kg_feat)
        inter_tensor = self._dataframe_to_interaction(self.inter_feat)

        head_entity = kg_tensor[self.head_entity_field] + user_num
        tail_entity = kg_tensor[self.tail_entity_field] + user_num

        user = inter_tensor[self.uid_field]
        item = inter_tensor[self.iid_field] + user_num

        src = torch.cat([user, item, head_entity])
        tgt = torch.cat([item, user, tail_entity])

        if show_relation:
            ui_rel_num = user.shape[0]
            ui_rel_id = self.relation_num - 1
            assert self.field2id_token[self.relation_field][ui_rel_id] == '[UI-Relation]'
            kg_rel = kg_tensor[self.relation_field]
            ui_rel = torch.full((2 * ui_rel_num,), ui_rel_id, dtype=kg_rel.dtype)
            edge = torch.cat([ui_rel, kg_rel])

        if form == 'dgl':
            import dgl
            graph = dgl.graph((src, tgt))
            if show_relation:
                graph.edata[self.relation_field] = edge
            return graph
        elif form == 'pyg':
            from torch_geometric.data import Data
            edge_attr = edge if show_relation else None
            graph = Data(edge_index=torch.stack([src, tgt]), edge_attr=edge_attr)
            return graph
        else:
            raise NotImplementedError('graph format [{}] has not been implemented.'.format(form))

    def ckg_graph(self, form='coo', value_field=None):
        if value_field is not None and value_field != self.relation_field:
            raise ValueError('value_field [{}] can only be [{}] in ckg_graph.'.format(
                value_field, self.relation_field
            ))
        show_relation = value_field is not None

        if form in ['coo', 'csr']:
            return self._create_ckg_sparse_matrix(form, show_relation)
        elif form in ['dgl', 'pyg']:
            return self._create_ckg_graph(form, show_relation)
        else:
            raise NotImplementedError('ckg graph format [{}] has not been implemented.')
