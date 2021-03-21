# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/9, 2020/9/15, 2020/9/22
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.social_dataset
#############################
"""

import os

from recbole.data.dataset import Dataset
from recbole.data.utils import dlapi
from recbole.utils import FeatureSource
from recbole.utils.utils import set_color


class SocialDataset(Dataset):
    """:class:`SocialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and load ``.net`` additionally.

    It also provides several interfaces to transfer ``.net`` features into coo sparse matrix,
    csr sparse matrix, :class:`DGL.Graph` or :class:`PyG.Data`.

    Attributes:
        source_field (str): The same as ``config['SOURCE_ID_FIELD']``.

        target_field (str): The same as ``config['TARGET_ID_FIELD']``.

        net_feat (pandas.DataFrame): Internal data structure stores the network features.
            It's loaded from file ``.net``.
    """

    def __init__(self, config):
        super().__init__(config)

    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.source_field = self.config['SOURCE_ID_FIELD']
        self.target_field = self.config['TARGET_ID_FIELD']
        self._check_field('source_field', 'target_field')

        self.logger.debug(set_color('source_id_field', 'blue') + f': {self.source_field}')
        self.logger.debug(set_color('target_id_field', 'blue') + f': {self.target_field}')

    def _load_data(self, token, dataset_path):
        """Load ``.net`` additionally.
        """
        super()._load_data(token, dataset_path)
        self.net_feat = self._load_net(self.dataset_name, self.dataset_path)

    def _build_feat_name_list(self):
        feat_name_list = super()._build_feat_name_list()
        if self.net_feat is not None:
            feat_name_list.append('net_feat')
        return feat_name_list

    def _load_net(self, dataset_name, dataset_path):
        net_file_path = os.path.join(dataset_path, f'{dataset_name}.net')
        if os.path.isfile(net_file_path):
            net_feat = self._load_feat(net_file_path, FeatureSource.NET)
            if net_feat is None:
                raise ValueError('.net file exist, but net_feat is None, please check your load_col')
            return net_feat
        else:
            raise ValueError(f'File {net_file_path} not exist.')

    def _get_fields_in_same_space(self):
        """Parsing ``config['fields_in_same_space']``. See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            - Each field can only exist ONCE in ``config['fields_in_same_space']``.
            - user_id and item_id can not exist in ``config['fields_in_same_space']``.
            - only token-like fields can exist in ``config['fields_in_same_space']``.
            - ``source_id`` and ``target_id`` should be remapped with ``user_id``.
        """
        fields_in_same_space = super()._get_fields_in_same_space()
        fields_in_same_space = [
            _ for _ in fields_in_same_space if (self.source_field not in _) and (self.target_field not in _)
        ]
        for field_set in fields_in_same_space:
            if self.uid_field in field_set:
                field_set.update({self.source_field, self.target_field})

        return fields_in_same_space

    @dlapi.set()
    def net_graph(self, form='coo', value_field=None):
        """Get graph or sparse matrix that describe relations between users.

        For an edge of <src, tgt>, ``graph[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``graph[src, tgt] = self.net_feat[value_field][src, tgt]``.

        Currently, we support graph in `DGL`_ and `PyG`_,
        and two type of sparse matrices, ``coo`` and ``csr``.

        Args:
            form (str, optional): Format of sparse matrix, or library of graph data structure.
                Defaults to ``coo``.
            value_field (str, optional): edge attributes of graph, or data of sparse matrix,
                Defaults to ``None``.

        Returns:
            Graph / Sparse matrix of relations.

        .. _DGL:
            https://www.dgl.ai/

        .. _PyG:
            https://github.com/rusty1s/pytorch_geometric
        """
        args = [self.net_feat, self.source_field, self.target_field, form, value_field]
        if form in ['coo', 'csr']:
            return self._create_sparse_matrix(*args)
        elif form in ['dgl', 'pyg']:
            return self._create_graph(*args)
        else:
            raise NotImplementedError('net graph format [{}] has not been implemented.')

    def __str__(self):
        info = [super().__str__(), f'The number of connections of social network: {len(self.net_feat)}']
        return '\n'.join(info)
