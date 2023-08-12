# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/16, 2020/9/15, 2020/10/25, 2022/7/10
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen, Lanling Xu
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn, xulanling_sherry@163.com

"""
recbole.data.kg_dataset
##########################
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix

from recbole.data.dataset import Dataset
from recbole.utils import FeatureSource, FeatureType, set_color
from recbole.utils.url import decide_download, download_url, extract_zip


class KnowledgeBasedDataset(Dataset):
    """:class:`KnowledgeBasedDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and load ``.kg`` and ``.link`` additionally.

    Entities are remapped together with ``item_id`` specially.
    All entities are remapped into three consecutive ID sections.

    - virtual entities that only exist in interaction data.
    - entities that exist both in interaction data and kg triplets.
    - entities only exist in kg triplets.

    It also provides several interfaces to transfer ``.kg`` features into coo sparse matrix,
    csr sparse matrix, :class:`DGL.Graph` or :class:`PyG.Data`.

    Attributes:
        head_entity_field (str): The same as ``config['HEAD_ENTITY_ID_FIELD']``.

        tail_entity_field (str): The same as ``config['TAIL_ENTITY_ID_FIELD']``.

        relation_field (str): The same as ``config['RELATION_ID_FIELD']``.

        entity_field (str): The same as ``config['ENTITY_ID_FIELD']``.

        kg_feat (pandas.DataFrame): Internal data structure stores the kg triplets.
            It's loaded from file ``.kg``.

        item2entity (dict): Dict maps ``item_id`` to ``entity``,
            which is loaded from  file ``.link``.

        entity2item (dict): Dict maps ``entity`` to ``item_id``,
            which is loaded from  file ``.link``.

    Note:
        :attr:`entity_field` doesn't exist exactly. It's only a symbol,
        representing entity features.

        ``[UI-Relation]`` is a special relation token.
    """

    def __init__(self, config):
        super().__init__(config)

    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.head_entity_field = self.config["HEAD_ENTITY_ID_FIELD"]
        self.tail_entity_field = self.config["TAIL_ENTITY_ID_FIELD"]
        self.relation_field = self.config["RELATION_ID_FIELD"]
        self.entity_field = self.config["ENTITY_ID_FIELD"]
        self.kg_reverse_r = self.config["kg_reverse_r"]
        self.entity_kg_num_interval = self.config["entity_kg_num_interval"]
        self.relation_kg_num_interval = self.config["relation_kg_num_interval"]
        self._check_field(
            "head_entity_field", "tail_entity_field", "relation_field", "entity_field"
        )
        self.set_field_property(
            self.entity_field, FeatureType.TOKEN, FeatureSource.KG, 1
        )

        self.logger.debug(
            set_color("relation_field", "blue") + f": {self.relation_field}"
        )
        self.logger.debug(set_color("entity_field", "blue") + f": {self.entity_field}")

    def _data_filtering(self):
        super()._data_filtering()
        self._filter_kg_by_triple_num()
        self._filter_link()

    def _filter_kg_by_triple_num(self):
        """Filter by number of triples.

        The interval of the number of triples can be set, and only entities/relations
        whose number of triples is in the specified interval can be retained.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Lower bound of the interval is also called k-core filtering, which means this method
            will filter loops until all the entities and relations has at least k triples.
        """
        entity_kg_num_interval = self._parse_intervals_str(
            self.config["entity_kg_num_interval"]
        )
        relation_kg_num_interval = self._parse_intervals_str(
            self.config["relation_kg_num_interval"]
        )

        if entity_kg_num_interval is None and relation_kg_num_interval is None:
            return

        entity_kg_num = Counter()
        if entity_kg_num_interval:
            head_entity_kg_num = Counter(self.kg_feat[self.head_entity_field].values)
            tail_entity_kg_num = Counter(self.kg_feat[self.tail_entity_field].values)
            entity_kg_num = head_entity_kg_num + tail_entity_kg_num
        relation_kg_num = (
            Counter(self.kg_feat[self.relation_field].values)
            if relation_kg_num_interval
            else Counter()
        )

        while True:
            ban_head_entities = self._get_illegal_ids_by_inter_num(
                field=self.head_entity_field,
                feat=None,
                inter_num=entity_kg_num,
                inter_interval=entity_kg_num_interval,
            )
            ban_tail_entities = self._get_illegal_ids_by_inter_num(
                field=self.tail_entity_field,
                feat=None,
                inter_num=entity_kg_num,
                inter_interval=entity_kg_num_interval,
            )
            ban_entities = ban_head_entities | ban_tail_entities
            ban_relations = self._get_illegal_ids_by_inter_num(
                field=self.relation_field,
                feat=None,
                inter_num=relation_kg_num,
                inter_interval=relation_kg_num_interval,
            )
            if len(ban_entities) == 0 and len(ban_relations) == 0:
                break

            dropped_kg = pd.Series(False, index=self.kg_feat.index)
            head_entity_kg = self.kg_feat[self.head_entity_field]
            tail_entity_kg = self.kg_feat[self.tail_entity_field]
            relation_kg = self.kg_feat[self.relation_field]
            dropped_kg |= head_entity_kg.isin(ban_entities)
            dropped_kg |= tail_entity_kg.isin(ban_entities)
            dropped_kg |= relation_kg.isin(ban_relations)

            entity_kg_num -= Counter(head_entity_kg[dropped_kg].values)
            entity_kg_num -= Counter(tail_entity_kg[dropped_kg].values)
            relation_kg_num -= Counter(relation_kg[dropped_kg].values)

            dropped_index = self.kg_feat.index[dropped_kg]
            self.logger.debug(f"[{len(dropped_index)}] dropped triples.")
            self.kg_feat.drop(dropped_index, inplace=True)

    def _filter_link(self):
        """Filter rows of :attr:`item2entity` and :attr:`entity2item`,
        whose ``entity_id`` doesn't occur in kg triplets and
        ``item_id`` doesn't occur in interaction records.
        """
        item_tokens = self._get_rec_item_token()
        ent_tokens = self._get_entity_token()
        illegal_item = set()
        illegal_ent = set()
        for item in self.item2entity:
            ent = self.item2entity[item]
            if item not in item_tokens or ent not in ent_tokens:
                illegal_item.add(item)
                illegal_ent.add(ent)
        for item in illegal_item:
            del self.item2entity[item]
        for ent in illegal_ent:
            del self.entity2item[ent]
        remained_inter = pd.Series(True, index=self.inter_feat.index)
        remained_inter &= self.inter_feat[self.iid_field].isin(self.item2entity.keys())
        self.inter_feat.drop(self.inter_feat.index[~remained_inter], inplace=True)

    def _download(self):
        super()._download()

        url = self._get_download_url("kg_url", allow_none=True)
        if url is None:
            return
        self.logger.info(f"Prepare to download linked knowledge graph from [{url}].")

        if decide_download(url):
            # No need to create dir, as `super()._download()` has created one.
            path = download_url(url, self.dataset_path)
            extract_zip(path, self.dataset_path)
            os.unlink(path)
            self.logger.info(
                f"\nLinked KG for [{self.dataset_name}] requires additional conversion "
                f"to atomic files (.kg and .link).\n"
                f"Please refer to https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools#knowledge-aware-datasets "
                f"for detailed instructions.\n"
                f"You can run RecBole after the conversion, see you soon."
            )
            exit(0)
        else:
            self.logger.info("Stop download.")
            exit(-1)

    def _load_data(self, token, dataset_path):
        super()._load_data(token, dataset_path)
        self.kg_feat = self._load_kg(self.dataset_name, self.dataset_path)
        self.item2entity, self.entity2item = self._load_link(
            self.dataset_name, self.dataset_path
        )

    def __str__(self):
        info = [
            super().__str__(),
            f"The number of entities: {self.entity_num}",
            f"The number of relations: {self.relation_num}",
            f"The number of triples: {len(self.kg_feat)}",
            f"The number of items that have been linked to KG: {len(self.item2entity)}",
        ]  # yapf: disable
        return "\n".join(info)

    def _build_feat_name_list(self):
        feat_name_list = super()._build_feat_name_list()
        if self.kg_feat is not None:
            feat_name_list.append("kg_feat")
        return feat_name_list

    def _load_kg(self, token, dataset_path):
        self.logger.debug(set_color(f"Loading kg from [{dataset_path}].", "green"))
        kg_path = os.path.join(dataset_path, f"{token}.kg")
        if not os.path.isfile(kg_path):
            raise ValueError(f"[{token}.kg] not found in [{dataset_path}].")
        df = self._load_feat(kg_path, FeatureSource.KG)
        self._check_kg(df)
        return df

    def _check_kg(self, kg):
        kg_warn_message = "kg data requires field [{}]"
        assert self.head_entity_field in kg, kg_warn_message.format(
            self.head_entity_field
        )
        assert self.tail_entity_field in kg, kg_warn_message.format(
            self.tail_entity_field
        )
        assert self.relation_field in kg, kg_warn_message.format(self.relation_field)

    def _load_link(self, token, dataset_path):
        self.logger.debug(set_color(f"Loading link from [{dataset_path}].", "green"))
        link_path = os.path.join(dataset_path, f"{token}.link")
        if not os.path.isfile(link_path):
            raise ValueError(f"[{token}.link] not found in [{dataset_path}].")
        df = self._load_feat(link_path, "link")
        self._check_link(df)

        item2entity, entity2item = {}, {}
        for item_id, entity_id in zip(
            df[self.iid_field].values, df[self.entity_field].values
        ):
            item2entity[item_id] = entity_id
            entity2item[entity_id] = item_id
        return item2entity, entity2item

    def _check_link(self, link):
        link_warn_message = "link data requires field [{}]"
        assert self.entity_field in link, link_warn_message.format(self.entity_field)
        assert self.iid_field in link, link_warn_message.format(self.iid_field)

    def _init_alias(self):
        """Add :attr:`alias_of_entity_id`, :attr:`alias_of_relation_id` and update :attr:`_rest_fields`."""
        self._set_alias("entity_id", [self.head_entity_field, self.tail_entity_field])
        self._set_alias("relation_id", [self.relation_field])

        super()._init_alias()

        self._rest_fields = np.setdiff1d(
            self._rest_fields, [self.entity_field], assume_unique=True
        )

    def _get_rec_item_token(self):
        """Get set of entity tokens from fields in ``rec`` level."""
        remap_list = self._get_remap_list(self.alias["item_id"])
        tokens, _ = self._concat_remaped_tokens(remap_list)
        return set(tokens)

    def _get_entity_token(self):
        """Get set of entity tokens from fields in ``ent`` level."""
        remap_list = self._get_remap_list(self.alias["entity_id"])
        tokens, _ = self._concat_remaped_tokens(remap_list)
        return set(tokens)

    def _reset_ent_remapID(self, field, idmap, id2token, token2id):
        self.field2id_token[field] = id2token
        self.field2token_id[field] = token2id
        for feat in self.field2feats(field):
            ftype = self.field2type[field]
            if ftype == FeatureType.TOKEN:
                old_idx = feat[field].values
            else:
                old_idx = feat[field].agg(np.concatenate)

            new_idx = idmap[old_idx]

            if ftype == FeatureType.TOKEN:
                feat[field] = new_idx
            else:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_idx, split_point)

    def _merge_item_and_entity(self):
        """Merge item-id and entity-id into the same id-space."""
        item_token = self.field2id_token[self.iid_field]
        entity_token = self.field2id_token[self.head_entity_field]
        item_num = len(item_token)
        link_num = len(self.item2entity)
        entity_num = len(entity_token)

        # reset item id
        item_priority = np.array([token in self.item2entity for token in item_token])
        item_order = np.argsort(item_priority, kind="stable")
        item_id_map = np.zeros_like(item_order)
        item_id_map[item_order] = np.arange(item_num)
        new_item_id2token = item_token[item_order]
        new_item_token2id = {t: i for i, t in enumerate(new_item_id2token)}
        for field in self.alias["item_id"]:
            self._reset_ent_remapID(
                field, item_id_map, new_item_id2token, new_item_token2id
            )

        # reset entity id
        entity_priority = np.array(
            [
                token != "[PAD]" and token not in self.entity2item
                for token in entity_token
            ]
        )
        entity_order = np.argsort(entity_priority, kind="stable")
        entity_id_map = np.zeros_like(entity_order)
        for i in entity_order[1 : link_num + 1]:
            entity_id_map[i] = new_item_token2id[self.entity2item[entity_token[i]]]
        entity_id_map[entity_order[link_num + 1 :]] = np.arange(
            item_num, item_num + entity_num - link_num - 1
        )
        new_entity_id2token = np.concatenate(
            [new_item_id2token, entity_token[entity_order[link_num + 1 :]]]
        )
        for i in range(item_num - link_num, item_num):
            new_entity_id2token[i] = self.item2entity[new_entity_id2token[i]]
        new_entity_token2id = {t: i for i, t in enumerate(new_entity_id2token)}
        for field in self.alias["entity_id"]:
            self._reset_ent_remapID(
                field, entity_id_map, new_entity_id2token, new_entity_token2id
            )
        self.field2id_token[self.entity_field] = new_entity_id2token
        self.field2token_id[self.entity_field] = new_entity_token2id

    def _add_auxiliary_relation(self):
        """Add auxiliary relations in ``self.relation_field``."""
        if self.kg_reverse_r:
            # '0' is used for padding, so the number needs to be reduced by one
            original_rel_num = len(self.field2id_token[self.relation_field]) - 1
            original_hids = self.kg_feat[self.head_entity_field]
            original_tids = self.kg_feat[self.tail_entity_field]
            original_rels = self.kg_feat[self.relation_field]

            # Internal id gap of a relation and its reverse edge is original relation num
            reverse_rels = original_rels + original_rel_num

            # Add mapping for internal and external ID of relations
            for i in range(1, original_rel_num + 1):
                original_token = self.field2id_token[self.relation_field][i]
                reverse_token = original_token + "_r"
                self.field2token_id[self.relation_field][reverse_token] = (
                    i + original_rel_num
                )
                self.field2id_token[self.relation_field] = np.append(
                    self.field2id_token[self.relation_field], reverse_token
                )

            # Update knowledge graph triples with reverse relations
            reverse_kg_data = {
                self.head_entity_field: original_tids,
                self.relation_field: reverse_rels,
                self.tail_entity_field: original_hids,
            }
            reverse_kg_feat = pd.DataFrame(reverse_kg_data)
            self.kg_feat = pd.concat([self.kg_feat, reverse_kg_feat])

        # Add UI-relation pairs in the relation field
        kg_rel_num = len(self.field2id_token[self.relation_field])
        self.field2token_id[self.relation_field]["[UI-Relation]"] = kg_rel_num
        self.field2id_token[self.relation_field] = np.append(
            self.field2id_token[self.relation_field], "[UI-Relation]"
        )

    def _remap_ID_all(self):
        super()._remap_ID_all()
        self._merge_item_and_entity()
        self._add_auxiliary_relation()

    @property
    def relation_num(self):
        """Get the number of different tokens of ``self.relation_field``.

        Returns:
            int: Number of different tokens of ``self.relation_field``.
        """
        return self.num(self.relation_field)

    @property
    def entity_num(self):
        """Get the number of different tokens of entities, including virtual entities.

        Returns:
            int: Number of different tokens of entities, including virtual entities.
        """
        return self.num(self.entity_field)

    @property
    def head_entities(self):
        """
        Returns:
            numpy.ndarray: List of head entities of kg triplets.
        """
        return self.kg_feat[self.head_entity_field].numpy()

    @property
    def tail_entities(self):
        """
        Returns:
            numpy.ndarray: List of tail entities of kg triplets.
        """
        return self.kg_feat[self.tail_entity_field].numpy()

    @property
    def relations(self):
        """
        Returns:
            numpy.ndarray: List of relations of kg triplets.
        """
        return self.kg_feat[self.relation_field].numpy()

    @property
    def entities(self):
        """
        Returns:
            numpy.ndarray: List of entity id, including virtual entities.
        """
        return np.arange(self.entity_num)

    def kg_graph(self, form="coo", value_field=None):
        """Get graph or sparse matrix that describe relations between entities.

        For an edge of <src, tgt>, ``graph[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``graph[src, tgt] = self.kg_feat[value_field][src, tgt]``.

        Currently, we support graph in `DGL`_ and `PyG`_,
        and two type of sparse matrices, ``coo`` and ``csr``.

        Args:
            form (str, optional): Format of sparse matrix, or library of graph data structure.
                Defaults to ``coo``.
            value_field (str, optional): edge attributes of graph, or data of sparse matrix,
                Defaults to ``None``.

        Returns:
            Graph / Sparse matrix of kg triplets.

        .. _DGL:
            https://www.dgl.ai/

        .. _PyG:
            https://github.com/rusty1s/pytorch_geometric
        """
        args = [
            self.kg_feat,
            self.head_entity_field,
            self.tail_entity_field,
            form,
            value_field,
        ]
        if form in ["coo", "csr"]:
            return self._create_sparse_matrix(*args)
        elif form in ["dgl", "pyg"]:
            return self._create_graph(*args)
        else:
            raise NotImplementedError("kg graph format [{}] has not been implemented.")

    def _create_ckg_sparse_matrix(self, form="coo", show_relation=False):
        user_num = self.user_num

        hids = self.head_entities + user_num
        tids = self.tail_entities + user_num

        uids = self.inter_feat[self.uid_field].numpy()
        iids = self.inter_feat[self.iid_field].numpy() + user_num

        ui_rel_num = len(uids)
        ui_rel_id = self.relation_num - 1
        assert self.field2id_token[self.relation_field][ui_rel_id] == "[UI-Relation]"

        src = np.concatenate([uids, iids, hids])
        tgt = np.concatenate([iids, uids, tids])

        if not show_relation:
            data = np.ones(len(src))
        else:
            kg_rel = self.kg_feat[self.relation_field].numpy()
            ui_rel = np.full(2 * ui_rel_num, ui_rel_id, dtype=kg_rel.dtype)
            data = np.concatenate([ui_rel, kg_rel])
        node_num = self.entity_num + self.user_num
        mat = coo_matrix((data, (src, tgt)), shape=(node_num, node_num))
        if form == "coo":
            return mat
        elif form == "csr":
            return mat.tocsr()
        else:
            raise NotImplementedError(
                f"Sparse matrix format [{form}] has not been implemented."
            )

    def _create_ckg_graph(self, form="dgl", show_relation=False):
        user_num = self.user_num

        kg_tensor = self.kg_feat
        inter_tensor = self.inter_feat

        head_entity = kg_tensor[self.head_entity_field] + user_num
        tail_entity = kg_tensor[self.tail_entity_field] + user_num

        user = inter_tensor[self.uid_field]
        item = inter_tensor[self.iid_field] + user_num

        src = torch.cat([user, item, head_entity])
        tgt = torch.cat([item, user, tail_entity])

        if show_relation:
            ui_rel_num = user.shape[0]
            ui_rel_id = self.relation_num - 1
            assert (
                self.field2id_token[self.relation_field][ui_rel_id] == "[UI-Relation]"
            )
            kg_rel = kg_tensor[self.relation_field]
            ui_rel = torch.full((2 * ui_rel_num,), ui_rel_id, dtype=kg_rel.dtype)
            edge = torch.cat([ui_rel, kg_rel])

        if form == "dgl":
            import dgl

            graph = dgl.graph((src, tgt))
            if show_relation:
                graph.edata[self.relation_field] = edge
            return graph
        elif form == "pyg":
            from torch_geometric.data import Data

            edge_attr = edge if show_relation else None
            graph = Data(edge_index=torch.stack([src, tgt]), edge_attr=edge_attr)
            return graph
        else:
            raise NotImplementedError(
                f"Graph format [{form}] has not been implemented."
            )

    def ckg_graph(self, form="coo", value_field=None):
        """Get graph or sparse matrix that describe relations of CKG,
        which combines interactions and kg triplets into the same graph.

        Item ids and entity ids are added by ``user_num`` temporally.

        For an edge of <src, tgt>, ``graph[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``graph[src, tgt] = self.kg_feat[self.relation_field][src, tgt]``
        or ``graph[src, tgt] = [UI-Relation]``.

        Currently, we support graph in `DGL`_ and `PyG`_,
        and two type of sparse matrices, ``coo`` and ``csr``.

        Args:
            form (str, optional): Format of sparse matrix, or library of graph data structure.
                Defaults to ``coo``.
            value_field (str, optional): ``self.relation_field`` or ``None``,
                Defaults to ``None``.

        Returns:
            Graph / Sparse matrix of kg triplets.

        .. _DGL:
            https://www.dgl.ai/

        .. _PyG:
            https://github.com/rusty1s/pytorch_geometric
        """
        if value_field is not None and value_field != self.relation_field:
            raise ValueError(
                f"Value_field [{value_field}] can only be [{self.relation_field}] in ckg_graph."
            )
        show_relation = value_field is not None

        if form in ["coo", "csr"]:
            return self._create_ckg_sparse_matrix(form, show_relation)
        elif form in ["dgl", "pyg"]:
            return self._create_ckg_graph(form, show_relation)
        else:
            raise NotImplementedError("ckg graph format [{}] has not been implemented.")
