# @Time   : 2020/6/28
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/28 2020/10/13, 2020/11/10
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.dataset
##########################
"""

import copy
import json
import os
from collections import Counter
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from scipy.sparse import coo_matrix
from sklearn.impute import SimpleImputer

from recbole.utils import FeatureSource, FeatureType
from recbole.data.interaction import Interaction
from recbole.data.utils import dlapi


class Dataset(object):
    """:class:`Dataset` stores the original dataset in memory.
    It provides many useful functions for data preprocessing, such as k-core data filtering and missing value
    imputation. Features are stored as :class:`pandas.DataFrame` inside :class:`~recbole.data.dataset.dataset.Dataset`.
    General and Context-aware Models can use this class.

    By calling method :meth:`~recbole.data.dataset.dataset.Dataset.build()`, it will processing dataset into
    DataLoaders, according to :class:`~recbole.config.eval_setting.EvalSetting`.

    Args:
        config (Config): Global configuration object.
        saved_dataset (str, optional): Restore Dataset object from ``saved_dataset``. Defaults to ``None``.

    Attributes:
        dataset_name (str): Name of this dataset.

        dataset_path (str): Local file path of this dataset.

        field2type (dict): Dict mapping feature name (str) to its type (:class:`~recbole.utils.enum_type.FeatureType`).

        field2source (dict): Dict mapping feature name (str) to its source
            (:class:`~recbole.utils.enum_type.FeatureSource`).
            Specially, if feature is loaded from Arg ``additional_feat_suffix``, its source has type str,
            which is the suffix of its local file (also the suffix written in Arg ``additional_feat_suffix``).

        field2id_token (dict): Dict mapping feature name (str) to a list, which stores the original token of
            this feature. For example, if ``test`` is token-like feature, ``token_a`` is remapped to 1, ``token_b``
            is remapped to 2. Then ``field2id_token['test'] = ['[PAD]', 'token_a', 'token_b']``. (Note that 0 is
            always PADDING for token-like features.)

        field2seqlen (dict): Dict mapping feature name (str) to its sequence length (int).
            For sequence features, their length can be either set in config,
            or set to the max sequence length of this feature.
            For token and float features, their length is 1.

        uid_field (str or None): The same as ``config['USER_ID_FIELD']``.

        iid_field (str or None): The same as ``config['ITEM_ID_FIELD']``.

        label_field (str or None): The same as ``config['LABEL_FIELD']``.

        time_field (str or None): The same as ``config['TIME_FIELD']``.

        inter_feat (:class:`pandas.DataFrame`): Internal data structure stores the interaction features.
            It's loaded from file ``.inter``.

        user_feat (:class:`pandas.DataFrame` or None): Internal data structure stores the user features.
            It's loaded from file ``.user`` if existed.

        item_feat (:class:`pandas.DataFrame` or None): Internal data structure stores the item features.
            It's loaded from file ``.item`` if existed.

        feat_list (list): A list contains all the features (:class:`pandas.DataFrame`), including additional features.
    """
    def __init__(self, config, saved_dataset=None):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()
        self._dataloader_apis = {'field2type', 'field2source', 'field2id_token'}
        self._dataloader_apis.update(dlapi.dataloader_apis)

        if saved_dataset is None:
            self._from_scratch()
        else:
            self._restore_saved_dataset(saved_dataset)

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.debug('Loading {} from scratch'.format(self.__class__))

        self._get_preset()
        self._get_field_from_config()
        self._load_data(self.dataset_name, self.dataset_path)
        self._data_processing()

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        self.dataset_path = self.config['data_path']
        self._fill_nan_flag = self.config['fill_nan']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2seqlen = self.config['seq_len'] or {}
        self._preloaded_weight = {}
        self.benchmark_filename_list = self.config['benchmark_filename']

    def _get_field_from_config(self):
        """Initialization common field names.
        """
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        self.logger.debug('uid_field: {}'.format(self.uid_field))
        self.logger.debug('iid_field: {}'.format(self.iid_field))

    def _data_processing(self):
        """Data preprocessing, including:

        - K-core data filtering
        - Value-based data filtering
        - Remap ID
        - Missing value imputation
        - Normalization
        - Preloading weights initialization
        """
        self.feat_list = self._build_feat_list()
        if self.benchmark_filename_list is None:
            self._data_filtering()

        self._remap_ID_all()
        self._user_item_feat_preparation()
        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()
        self._preload_weight_matrix()

    def _data_filtering(self):
        """Data filtering

        - Filter missing user_id or item_id
        - Value-based data filtering
        - K-core data filtering

        Note:
            After filtering, feats(``DataFrame``) has non-continuous index,
            thus :meth:`~recbole.data.dataset.dataset.Dataset._reset_index()` will reset the index of feats.
        """
        self._filter_nan_user_or_item()
        self._remove_duplication()
        self._filter_by_field_value()
        self._filter_by_inter_num()
        self._reset_index()

    def _build_feat_list(self):
        """Feat list building.

        Any feat loaded by Dataset can be found in ``feat_list``

        Returns:
            builded feature list.

        Note:
            Subclasses can inherit this method to add new feat.
        """
        feat_list = [feat for feat in [self.inter_feat, self.user_feat, self.item_feat] if feat is not None]
        if self.config['additional_feat_suffix'] is not None:
            for suf in self.config['additional_feat_suffix']:
                if hasattr(self, '{}_feat'.format(suf)):
                    feat_list.append(getattr(self, '{}_feat'.format(suf)))
        return feat_list

    def _restore_saved_dataset(self, saved_dataset):
        """Restore saved dataset from ``saved_dataset``.

        Args:
            saved_dataset (str): path for the saved dataset.
        """
        self.logger.debug('Restoring dataset from [{}]'.format(saved_dataset))

        if (saved_dataset is None) or (not os.path.isdir(saved_dataset)):
            raise ValueError('filepath [{}] need to be a dir'.format(saved_dataset))

        with open(os.path.join(saved_dataset, 'basic-info.json')) as file:
            basic_info = json.load(file)

        for k in basic_info:
            setattr(self, k, basic_info[k])

        feats = ['inter', 'user', 'item']
        for name in feats:
            cur_file_name = os.path.join(saved_dataset, '{}.csv'.format(name))
            if os.path.isfile(cur_file_name):
                df = pd.read_csv(cur_file_name)
                setattr(self, '{}_feat'.format(name), df)
            else:
                setattr(self, '{}_feat'.format(name), None)

        self._get_field_from_config()

    def _load_data(self, token, dataset_path):
        """Load features.

        Firstly load interaction features, then user/item features optionally,
        finally load additional features if ``config['additional_feat_suffix']`` is set.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.USER, 'uid_field')
        self.item_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.ITEM, 'iid_field')
        self._load_additional_feat(token, dataset_path)

    def _load_inter_feat(self, token, dataset_path):
        """Load interaction features.

        If ``config['benchmark_filename']`` is not set, load interaction features from ``.inter``.

        Otherwise, load interaction features from a file list, named ``dataset_name.xxx.inter``,
        where ``xxx`` if from ``config['benchmark_filename']``.
        After loading, ``self.file_size_list`` stores the length of each interaction file.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        if self.benchmark_filename_list is None:
            inter_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'inter'))
            if not os.path.isfile(inter_feat_path):
                raise ValueError('File {} not exist'.format(inter_feat_path))

            inter_feat = self._load_feat(inter_feat_path, FeatureSource.INTERACTION)
            self.logger.debug('interaction feature loaded successfully from [{}]'.format(inter_feat_path))
            self.inter_feat = inter_feat
        else:
            sub_inter_lens = []
            sub_inter_feats = []
            for filename in self.benchmark_filename_list:
                file_path = os.path.join(dataset_path, '{}.{}.{}'.format(token, filename, 'inter'))
                if os.path.isfile(file_path):
                    temp = self._load_feat(file_path, FeatureSource.INTERACTION)
                    sub_inter_feats.append(temp)
                    sub_inter_lens.append(len(temp))
                else:
                    raise ValueError('File {} not exist'.format(file_path))
            inter_feat = pd.concat(sub_inter_feats)
            self.inter_feat, self.file_size_list = inter_feat, sub_inter_lens

    def _load_user_or_item_feat(self, token, dataset_path, source, field_name):
        """Load user/item features.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
            source (FeatureSource): source of user/item feature.
            field_name (str): ``uid_field`` or ``iid_field``

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            ``user_id`` and ``item_id`` has source :obj:`~recbole.utils.enum_type.FeatureSource.USER_ID` and
            :obj:`~recbole.utils.enum_type.FeatureSource.ITEM_ID`
        """
        feat_path = os.path.join(dataset_path, '{}.{}'.format(token, source.value))
        if os.path.isfile(feat_path):
            feat = self._load_feat(feat_path, source)
            self.logger.debug('[{}] feature loaded successfully from [{}]'.format(source.value, feat_path))
        else:
            feat = None
            self.logger.debug('[{}] not found, [{}] features are not loaded'.format(feat_path, source.value))

        field = getattr(self, field_name, None)
        if feat is not None and field is None:
            raise ValueError('{} must be exist if {}_feat exist'.format(field_name, source.value))
        if feat is not None and field not in feat:
            raise ValueError('{} must be loaded if {}_feat is loaded'.format(field_name, source.value))

        if field in self.field2source:
            self.field2source[field] = FeatureSource(source.value + '_id')
        return feat

    def _load_additional_feat(self, token, dataset_path):
        """Load additional features.

        For those additional features, e.g. pretrained entity embedding, user can set them
        as ``config['additional_feat_suffix']``, then they will be loaded and stored in
        :attr:`feat_list`. See :doc:`../user_guide/data/data_args` for details.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        if self.config['additional_feat_suffix'] is None:
            return
        for suf in self.config['additional_feat_suffix']:
            if hasattr(self, '{}_feat'.format(suf)):
                raise ValueError('{}_feat already exist'.format(suf))
            feat_path = os.path.join(dataset_path, '{}.{}'.format(token, suf))
            if os.path.isfile(feat_path):
                feat = self._load_feat(feat_path, suf)
            else:
                raise ValueError('Additional feature file [{}] not found'.format(feat_path))
            setattr(self, '{}_feat'.format(suf), feat)

    def _get_load_and_unload_col(self, source):
        """Parsing ``config['load_col']`` and ``config['unload_col']`` according to source.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Args:
            source (FeatureSource): source of input file.

        Returns:
            tuple: tuple of parsed ``load_col`` and ``unload_col``, see :doc:`../user_guide/data/data_args` for details.
        """
        if isinstance(source, FeatureSource):
            source = source.value
        if self.config['load_col'] is None:
            load_col = None
        elif source not in self.config['load_col']:
            load_col = set()
        elif self.config['load_col'][source] == '*':
            load_col = None
        else:
            load_col = set(self.config['load_col'][source])

        if self.config['unload_col'] is not None and source in self.config['unload_col']:
            unload_col = set(self.config['unload_col'][source])
        else:
            unload_col = None

        if load_col and unload_col:
            raise ValueError('load_col [{}] and unload_col [{}] can not be set the same time'.format(
                load_col, unload_col))

        self.logger.debug('\n [{}]:\n\t load_col: [{}]\n\t unload_col: [{}]\n'.format(source, load_col, unload_col))
        return load_col, unload_col

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cutted off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        self.logger.debug('loading feature from [{}] (source: [{}])'.format(filepath, source))

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config['field_separator']
        columns = []
        usecols = []
        dtype = {}
        with open(filepath, 'r') as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(':')
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError('Type {} from field {} is not supported'.format(ftype, field))
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != 'link':
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith('seq'):
                    self.field2seqlen[field] = 1
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning('no columns has been loaded from [{}]'.format(source))
            return None

        df = pd.read_csv(filepath, delimiter=self.config['field_separator'], usecols=usecols, dtype=dtype)
        df.columns = columns

        seq_separator = self.config['seq_separator']
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith('seq'):
                continue
            df[field].fillna(value='0', inplace=True)
            if ftype == FeatureType.TOKEN_SEQ:
                df[field] = [list(filter(None, _.split(seq_separator))) for _ in df[field].values]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [list(map(float, filter(None, _.split(seq_separator)))) for _ in df[field].values]
            self.field2seqlen[field] = max(map(len, df[field].values))
        return df

    def _user_item_feat_preparation(self):
        """Sort :attr:`user_feat` and :attr:`item_feat` by ``user_id`` or ``item_id``.
        Missing values will be filled.
        """
        flag = False
        if self.user_feat is not None:
            new_user_df = pd.DataFrame({self.uid_field: np.arange(self.user_num)})
            self.user_feat = pd.merge(new_user_df, self.user_feat, on=self.uid_field, how='left')
            flag = True
            self.logger.debug('ordering user features by user id.')
        if self.item_feat is not None:
            new_item_df = pd.DataFrame({self.iid_field: np.arange(self.item_num)})
            self.item_feat = pd.merge(new_item_df, self.item_feat, on=self.iid_field, how='left')
            flag = True
            self.logger.debug('ordering item features by user id.')
        if flag:
            # CANNOT be removed
            # user/item feat has been updated, thus feat_list should be updated too.
            self.feat_list = self._build_feat_list()
            self._fill_nan_flag = True

    def _preload_weight_matrix(self):
        """Transfer preload weight features into :class:`numpy.ndarray` with shape ``[id_token_length]``
        or ``[id_token_length, seqlen]``. See :doc:`../user_guide/data/data_args` for detail arg setting.
        """
        preload_fields = self.config['preload_weight']
        if preload_fields is None:
            return
        drop_flag = self.config['drop_preload_weight']
        if drop_flag is None:
            drop_flag = True

        self.logger.debug('preload weight matrix for {}, drop=[{}]'.format(preload_fields, drop_flag))

        for preload_id_field in preload_fields:
            preload_value_field = preload_fields[preload_id_field]
            if preload_id_field not in self.field2source:
                raise ValueError('prelaod id field [{}] not exist'.format(preload_id_field))
            if preload_value_field not in self.field2source:
                raise ValueError('prelaod value field [{}] not exist'.format(preload_value_field))
            pid_source = self.field2source[preload_id_field]
            pv_source = self.field2source[preload_value_field]
            if pid_source != pv_source:
                raise ValueError('preload id field [{}] is from source [{}],'
                                 'while prelaod value field [{}] is from source [{}], which should be the same'.format(
                                     preload_id_field, pid_source, preload_value_field, pv_source
                                 ))
            for feat in self.feat_list:
                if preload_id_field in feat:
                    id_ftype = self.field2type[preload_id_field]
                    if id_ftype != FeatureType.TOKEN:
                        raise ValueError('prelaod id field [{}] should be type token, but is [{}]'.format(
                            preload_id_field, id_ftype
                        ))
                    value_ftype = self.field2type[preload_value_field]
                    token_num = self.num(preload_id_field)
                    if value_ftype == FeatureType.FLOAT:
                        matrix = np.zeros(token_num)
                        preload_ids = feat[preload_id_field].values
                        preload_values = feat[preload_value_field].values
                        for pid, pv in zip(preload_ids, preload_values):
                            matrix[pid] = pv
                    elif value_ftype == FeatureType.FLOAT_SEQ:
                        max_len = self.field2seqlen[preload_value_field]
                        matrix = np.zeros((token_num, max_len))
                        preload_ids = feat[preload_id_field].values
                        preload_values = feat[preload_value_field].to_list()
                        for pid, prow in zip(preload_ids, preload_values):
                            length = len(prow)
                            if length <= max_len:
                                matrix[pid, :length] = prow
                            else:
                                matrix[pid] = prow[:max_len]
                    else:
                        self.logger.warning('Field [{}] with type [{}] is not \'float\' or \'float_seq\', \
                                             which will not be handled by preload matrix.'.format(preload_value_field,
                                                                                                  value_ftype))
                        continue
                    self._preloaded_weight[preload_id_field] = matrix
                    if drop_flag:
                        self._del_col(preload_id_field)
                        self._del_col(preload_value_field)

    def _fill_nan(self):
        """Missing value imputation.

        For fields with type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN`, missing value will be filled by
        ``[PAD]``, which indexed as 0.

        For fields with type :obj:`~recbole.utils.enum_type.FeatureType.FLOAT`, missing value will be filled by
        the average of original data.

        For sequence features, missing value will be filled by ``[0]``. 
        """
        self.logger.debug('Filling nan')

        if not self._fill_nan_flag:
            return

        most_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent', copy=False)
        aveg = SimpleImputer(missing_values=np.nan, strategy='mean', copy=False)

        for feat in self.feat_list:
            for field in feat:
                ftype = self.field2type[field]
                if ftype == FeatureType.TOKEN:
                    feat[field] = most_freq.fit_transform(feat[field].values.reshape(-1, 1))
                elif ftype == FeatureType.FLOAT:
                    feat[field] = aveg.fit_transform(feat[field].values.reshape(-1, 1))
                elif ftype.value.endswith('seq'):
                    feat[field] = feat[field].apply(lambda x: [0]
                                                    if (not isinstance(x, np.ndarray) and (not isinstance(x, list)))
                                                    else x)

    def _normalize(self):
        """Normalization if ``config['normalize_field']`` or ``config['normalize_all']`` is set.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        .. math::
            x' = \frac{x - x_{min}}{x_{max} - x_{min}}

        Note:
            Only float-like fields can be normalized.
        """
        if self.config['normalize_field'] is not None and self.config['normalize_all'] is not None:
            raise ValueError('normalize_field and normalize_all can\'t be set at the same time')

        if self.config['normalize_field']:
            fields = self.config['normalize_field']
            for field in fields:
                ftype = self.field2type[field]
                if field not in self.field2type:
                    raise ValueError('Field [{}] doesn\'t exist'.format(field))
                elif ftype != FeatureType.FLOAT and ftype != FeatureType.FLOAT_SEQ:
                    self.logger.warning('{} is not a FLOAT/FLOAT_SEQ feat, which will not be normalized.'.format(field))
        elif self.config['normalize_all']:
            fields = self.float_like_fields
        else:
            return

        self.logger.debug('Normalized fields: {}'.format(fields))

        for feat in self.feat_list:
            for field in feat:
                if field not in fields:
                    continue
                ftype = self.field2type[field]
                if ftype == FeatureType.FLOAT:
                    lst = feat[field].values
                    mx, mn = max(lst), min(lst)
                    if mx == mn:
                        raise ValueError('All the same value in [{}] from [{}_feat]'.format(field, feat))
                    feat[field] = (lst - mn) / (mx - mn)
                elif ftype == FeatureType.FLOAT_SEQ:
                    split_point = np.cumsum(feat[field].agg(len))[:-1]
                    lst = feat[field].agg(np.concatenate)
                    mx, mn = max(lst), min(lst)
                    if mx == mn:
                        raise ValueError('All the same value in [{}] from [{}_feat]'.format(field, feat))
                    lst = (lst - mn) / (mx - mn)
                    lst = np.split(lst, split_point)
                    feat[field] = lst

    def _filter_nan_user_or_item(self):
        """Filter NaN user_id and item_id
        """
        for field, name in zip([self.uid_field, self.iid_field], ['user', 'item']):
            feat = getattr(self, name + '_feat')
            if feat is not None:
                dropped_feat = feat.index[feat[field].isnull()]
                if dropped_feat.any():
                    self.logger.warning('In {}_feat, line {}, {} do not exist, so they will be removed'.format(
                        name, list(dropped_feat + 2), field))
                    feat.drop(feat.index[dropped_feat], inplace=True)
            if field is not None:
                dropped_inter = self.inter_feat.index[self.inter_feat[field].isnull()]
                if dropped_inter.any():
                    self.logger.warning('In inter_feat, line {}, {} do not exist, so they will be removed'.format(
                        name, list(dropped_inter + 2), field))
                    self.inter_feat.drop(self.inter_feat.index[dropped_inter], inplace=True)

    def _remove_duplication(self):
        """Remove duplications in inter_feat.

        If :attr:`self.config['rm_dup_inter']` is not ``None``, it will remove duplicated user-item interactions.

        Note:
            Before removing duplicated user-item interactions, if :attr:`time_field` existed, :attr:`inter_feat`
            will be sorted by :attr:`time_field` in ascending order.
        """
        keep = self.config['rm_dup_inter']
        if keep is None:
            return
        self._check_field('uid_field', 'iid_field')

        if self.time_field in self.inter_feat:
            self.inter_feat.sort_values(by=[self.time_field], ascending=True, inplace=True)
            self.logger.info('Records in original dataset have been sorted by value of [{}] in ascending order.'.format(
                self.time_field))
        else:
            self.logger.warning('Timestamp field has not been loaded or specified, '
                                'thus strategy [{}] of duplication removal may be meaningless.'.format(keep))
        self.inter_feat.drop_duplicates(subset=[self.uid_field, self.iid_field], keep=keep, inplace=True)

    def _filter_by_inter_num(self):
        """Filter by number of interaction.

        Upper/Lower bounds can be set, only users/items between upper/lower bounds can be remained.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Lower bound is also called k-core filtering, which means this method will filter loops
            until all the users and items has at least k interactions.
        """
        while True:
            ban_users = self._get_illegal_ids_by_inter_num(field=self.uid_field, feat=self.user_feat,
                                                           max_num=self.config['max_user_inter_num'],
                                                           min_num=self.config['min_user_inter_num'])
            ban_items = self._get_illegal_ids_by_inter_num(field=self.iid_field, feat=self.item_feat,
                                                           max_num=self.config['max_item_inter_num'],
                                                           min_num=self.config['min_item_inter_num'])

            if len(ban_users) == 0 and len(ban_items) == 0:
                return

            if self.user_feat is not None:
                dropped_user = self.user_feat[self.uid_field].isin(ban_users)
                self.user_feat.drop(self.user_feat.index[dropped_user], inplace=True)

            if self.item_feat is not None:
                dropped_item = self.item_feat[self.iid_field].isin(ban_items)
                self.item_feat.drop(self.item_feat.index[dropped_item], inplace=True)

            dropped_inter = pd.Series(False, index=self.inter_feat.index)
            if self.uid_field:
                dropped_inter |= self.inter_feat[self.uid_field].isin(ban_users)
            if self.iid_field:
                dropped_inter |= self.inter_feat[self.iid_field].isin(ban_items)
            self.logger.debug('[{}] dropped interactions'.format(len(dropped_inter)))
            self.inter_feat.drop(self.inter_feat.index[dropped_inter], inplace=True)

    def _get_illegal_ids_by_inter_num(self, field, feat, max_num=None, min_num=None):
        """Given inter feat, return illegal ids, whose inter num out of [min_num, max_num]

        Args:
            field (str): field name of user_id or item_id.
            feat (pandas.DataFrame): interaction feature.
            max_num (int, optional): max number of interaction. Defaults to ``None``.
            min_num (int, optional): min number of interaction. Defaults to ``None``.

        Returns:
            set: illegal ids, whose inter num out of [min_num, max_num]
        """
        self.logger.debug('\n get_illegal_ids_by_inter_num:\n\t field=[{}], max_num=[{}], min_num=[{}]'.format(
            field, max_num, min_num
        ))

        if field is None:
            return set()
        if max_num is None and min_num is None:
            return set()

        max_num = max_num or np.inf
        min_num = min_num or -1

        ids = self.inter_feat[field].values
        inter_num = Counter(ids)
        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}

        if feat is not None:
            for id_ in feat[field].values:
                if inter_num[id_] < min_num:
                    ids.add(id_)
        self.logger.debug('[{}] illegal_ids_by_inter_num, field=[{}]'.format(len(ids), field))
        return ids

    def _filter_by_field_value(self):
        """Filter features according to its values.
        """
        filter_field = []
        filter_field += self._drop_by_value(self.config['lowest_val'], lambda x, y: x < y)
        filter_field += self._drop_by_value(self.config['highest_val'], lambda x, y: x > y)
        filter_field += self._drop_by_value(self.config['equal_val'], lambda x, y: x != y)
        filter_field += self._drop_by_value(self.config['not_equal_val'], lambda x, y: x == y)

        if not filter_field:
            return
        if self.config['drop_filter_field']:
            for field in set(filter_field):
                self._del_col(field)

    def _reset_index(self):
        """Reset index for all feats in :attr:`feat_list`.
        """
        for feat in self.feat_list:
            if feat.empty:
                raise ValueError('Some feat is empty, please check the filtering settings.')
            feat.reset_index(drop=True, inplace=True)

    def _drop_by_value(self, val, cmp):
        """Drop illegal rows by value.

        Args:
            val (float): value that compared to.
            cmp (function): return False if a row need to be droped

        Returns:
            field names that used to compare with val.
        """
        if val is None:
            return []

        self.logger.debug('drop_by_value: val={}'.format(val))
        filter_field = []
        for field in val:
            if field not in self.field2type:
                raise ValueError('field [{}] not defined in dataset'.format(field))
            if self.field2type[field] not in {FeatureType.FLOAT, FeatureType.FLOAT_SEQ}:
                raise ValueError('field [{}] is not float-like field in dataset, which can\'t be filter'.format(field))
            for feat in self.feat_list:
                if field in feat:
                    feat.drop(feat.index[cmp(feat[field].values, val[field])], inplace=True)
            filter_field.append(field)
        return filter_field

    def _del_col(self, field):
        """Delete columns

        Args:
            field (str): field name to be droped.
        """
        self.logger.debug('delete column [{}]'.format(field))
        for feat in self.feat_list:
            if field in feat:
                feat.drop(columns=field, inplace=True)
        for dct in [self.field2id_token, self.field2seqlen, self.field2source, self.field2type]:
            if field in dct:
                del dct[field]

    def _set_label_by_threshold(self):
        """Generate 0/1 labels according to value of features.

        According to ``config['threshold']``, those rows with value lower than threshold will
        be given negative label, while the other will be given positive label.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Key of ``config['threshold']`` if a field name.
            This field will be droped after label generation.
        """
        threshold = self.config['threshold']
        if threshold is None:
            return

        self.logger.debug('set label by {}'.format(threshold))

        if len(threshold) != 1:
            raise ValueError('threshold length should be 1')

        self.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        for field, value in threshold.items():
            if field in self.inter_feat:
                self.inter_feat[self.label_field] = (self.inter_feat[field] >= value).astype(int)
            else:
                raise ValueError('field [{}] not in inter_feat'.format(field))
            self._del_col(field)

    def _get_fields_in_same_space(self):
        """Parsing ``config['fields_in_same_space']``. See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            - Each field can only exist ONCE in ``config['fields_in_same_space']``.
            - user_id and item_id can not exist in ``config['fields_in_same_space']``.
            - only token-like fields can exist in ``config['fields_in_same_space']``.
        """
        fields_in_same_space = self.config['fields_in_same_space'] or []
        fields_in_same_space = [set(_) for _ in fields_in_same_space]
        additional = []
        token_like_fields = self.token_like_fields
        for field in token_like_fields:
            count = 0
            for field_set in fields_in_same_space:
                if field in field_set:
                    count += 1
            if count == 0:
                additional.append({field})
            elif count == 1:
                continue
            else:
                raise ValueError('field [{}] occurred in `fields_in_same_space` more than one time'.format(field))

        for field_set in fields_in_same_space:
            if self.uid_field in field_set and self.iid_field in field_set:
                raise ValueError('uid_field and iid_field can\'t in the same ID space')
            for field in field_set:
                if field not in token_like_fields:
                    raise ValueError('field [{}] is not a token-like field'.format(field))

        fields_in_same_space.extend(additional)
        return fields_in_same_space

    def _get_remap_list(self, field_set):
        """Transfer set of fields in the same remapping space into remap list.

        If ``uid_field`` or ``iid_field`` in ``field_set``,
        field in :attr:`inter_feat` will be remapped firstly,
        then field in :attr:`user_feat` or :attr:`item_feat` will be remapped next, finally others.

        Args:
            field_set (set): Set of fields in the same remapping space

        Returns:
            list:
            - feat (pandas.DataFrame)
            - field (str)
            - ftype (FeatureType)

            They will be concatenated in order, and remapped together.
        """
        remap_list = []
        for field, feat in zip([self.uid_field, self.iid_field], [self.user_feat, self.item_feat]):
            if field in field_set:
                field_set.remove(field)
                remap_list.append((self.inter_feat, field, FeatureType.TOKEN))
                if feat is not None:
                    remap_list.append((feat, field, FeatureType.TOKEN))
        for field in field_set:
            source = self.field2source[field]
            if isinstance(source, FeatureSource):
                source = source.value
            feat = getattr(self, '{}_feat'.format(source))
            ftype = self.field2type[field]
            remap_list.append((feat, field, ftype))
        return remap_list

    def _remap_ID_all(self):
        """Get ``config['fields_in_same_space']`` firstly, and remap each.
        """
        fields_in_same_space = self._get_fields_in_same_space()
        self.logger.debug('fields_in_same_space: {}'.format(fields_in_same_space))
        for field_set in fields_in_same_space:
            remap_list = self._get_remap_list(field_set)
            self._remap(remap_list)

    def _concat_remaped_tokens(self, remap_list):
        """Given ``remap_list``, concatenate values in order.

        Args:
            remap_list (list): See :meth:`_get_remap_list` for detail.

        Returns:
            tuple: tuple of:
            - tokens after concatenation.
            - split points that can be used to restore the concatenated tokens.
        """
        tokens = []
        for feat, field, ftype in remap_list:
            if ftype == FeatureType.TOKEN:
                tokens.append(feat[field].values)
            elif ftype == FeatureType.TOKEN_SEQ:
                tokens.append(feat[field].agg(np.concatenate))
        split_point = np.cumsum(list(map(len, tokens)))[:-1]
        tokens = np.concatenate(tokens)
        return tokens, split_point

    def _remap(self, remap_list):
        """Remap tokens using :meth:`pandas.factorize`.

        Args:
            remap_list (list): See :meth:`_get_remap_list` for detail.
        """
        tokens, split_point = self._concat_remaped_tokens(remap_list)
        new_ids_list, mp = pd.factorize(tokens)
        new_ids_list = np.split(new_ids_list + 1, split_point)
        mp = ['[PAD]'] + list(mp)

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if (field not in self.field2id_token):
                self.field2id_token[field] = mp
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    @dlapi.set()
    def num(self, field):
        """Given ``field``, for token-like fields, return the number of different tokens after remapping,
        for float-like fields, return ``1``.

        Args:
            field (str): field name to get token number.

        Returns:
            int: The number of different tokens (``1`` if ``field`` is a float-like field).
        """
        if field not in self.field2type:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        if self.field2type[field] not in {FeatureType.TOKEN, FeatureType.TOKEN_SEQ}:
            return self.field2seqlen[field]
        else:
            return len(self.field2id_token[field])

    @dlapi.set()
    def fields(self, ftype=None):
        """Given type of features, return all the field name of this type.
        if ``ftype = None``, return all the fields.

        Args:
            ftype (FeatureType, optional): Type of features.

        Returns:
            list: List of field names.
        """
        ftype = set(ftype) if ftype is not None else set(FeatureType)
        ret = []
        for field in self.field2type:
            tp = self.field2type[field]
            if tp in ftype:
                ret.append(field)
        return ret

    @property
    def float_like_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.FLOAT` and
        :obj:`~recbole.utils.enum_type.FeatureType.FLOAT_SEQ`.

        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.FLOAT, FeatureType.FLOAT_SEQ])

    @property
    def token_like_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN` and
        :obj:`~recbole.utils.enum_type.FeatureType.TOKEN_SEQ`.

        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.TOKEN, FeatureType.TOKEN_SEQ])

    @property
    def seq_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN_SEQ` and
        :obj:`~recbole.utils.enum_type.FeatureType.FLOAT_SEQ`.

        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.FLOAT_SEQ, FeatureType.TOKEN_SEQ])

    @property
    def non_seq_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN` and
        :obj:`~recbole.utils.enum_type.FeatureType.FLOAT`.

        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.FLOAT, FeatureType.TOKEN])

    def set_field_property(self, field, field_type, field_source, field_seqlen):
        """Set a new field's properties.

        Args:
            field (str): Name of the new field.
            field_type (FeatureType): Type of the new field.
            field_source (FeatureSource): Source of the new field.
            field_seqlen (int): max length of the sequence in ``field``.
                ``1`` if ``field``'s type is not sequence-like.
        """
        self.field2type[field] = field_type
        self.field2source[field] = field_source
        self.field2seqlen[field] = field_seqlen

    def copy_field_property(self, dest_field, source_field):
        """Copy properties from ``dest_field`` towards ``source_field``.

        Args:
            dest_field (str): Destination field.
            source_field (str): Source field.
        """
        self.field2type[dest_field] = self.field2type[source_field]
        self.field2source[dest_field] = self.field2source[source_field]
        self.field2seqlen[dest_field] = self.field2seqlen[source_field]

    @property
    @dlapi.set()
    def user_num(self):
        """Get the number of different tokens of ``self.uid_field``.

        Returns:
            int: Number of different tokens of ``self.uid_field``.
        """
        self._check_field('uid_field')
        return self.num(self.uid_field)

    @property
    @dlapi.set()
    def item_num(self):
        """Get the number of different tokens of ``self.iid_field``.

        Returns:
            int: Number of different tokens of ``self.iid_field``.
        """
        self._check_field('iid_field')
        return self.num(self.iid_field)

    @property
    def inter_num(self):
        """Get the number of interaction records.

        Returns:
            int: Number of interaction records.
        """
        return len(self.inter_feat)

    @property
    def avg_actions_of_users(self):
        """Get the average number of users' interaction records.

        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        return np.mean(self.inter_feat.groupby(self.uid_field).size())

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.

        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        return np.mean(self.inter_feat.groupby(self.iid_field).size())

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num

    @property
    def uid2index(self):
        """Sort ``self.inter_feat``,
        and get the mapping of user_id and index of its interaction records.

        Returns:
            tuple:
            - ``numpy.ndarray`` of tuple ``(uid, slice)``,
              interaction records between slice are all belong to the same uid.
            - ``numpy.ndarray`` of int,
              representing number of interaction records of each user.
        """
        self._check_field('uid_field')
        self.sort(by=self.uid_field, ascending=True)
        uid_list = []
        start, end = dict(), dict()
        for i, uid in enumerate(self.inter_feat[self.uid_field].values):
            if uid not in start:
                uid_list.append(uid)
                start[uid] = i
            end[uid] = i
        index = [(uid, slice(start[uid], end[uid] + 1)) for uid in uid_list]
        uid2items_num = [end[uid] - start[uid] + 1 for uid in uid_list]
        return np.array(index), np.array(uid2items_num)

    def _check_field(self, *field_names):
        """Given a name of attribute, check if it's exist.

        Args:
            *field_names (str): Fields to be checked.
        """
        for field_name in field_names:
            if getattr(self, field_name, None) is None:
                raise ValueError('{} isn\'t set'.format(field_name))

    def join(self, df):
        """Given interaction feature, join user/item feature into it.

        Args:
            df (pandas.DataFrame): Interaction feature to be joint.

        Returns:
            pandas.DataFrame: Interaction feature after joining operation.
        """
        if self.user_feat is not None and self.uid_field in df:
            df = pd.merge(df, self.user_feat, on=self.uid_field, how='left', suffixes=('_inter', '_user'))
        if self.item_feat is not None and self.iid_field in df:
            df = pd.merge(df, self.item_feat, on=self.iid_field, how='left', suffixes=('_inter', '_item'))
        return df

    def __getitem__(self, index, join=True):
        df = self.inter_feat[index]
        return self.join(df) if join else df

    def __len__(self):
        return len(self.inter_feat)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        if self.uid_field:
            info.extend(['The number of users: {}'.format(self.user_num),
                         'Average actions of users: {}'.format(self.avg_actions_of_users)])
        if self.iid_field:
            info.extend(['The number of items: {}'.format(self.item_num),
                         'Average actions of items: {}'.format(self.avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            info.append('The sparsity of the dataset: {}%'.format(self.sparsity * 100))
        info.append('Remain Fields: {}'.format(list(self.field2type)))
        return '\n'.join(info)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (pandas.DataFrame): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def _calcu_split_ids(self, tot, ratios):
        """Given split ratios, and total number, calculate the number of each part after splitting.

        Other than the first one, each part is rounded down.

        Args:
            tot (int): Total number.
            ratios (list): List of split ratios. No need to be normalized.

        Returns:
            list: Number of each part after splitting.
        """
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def split_by_ratio(self, ratios, group_by=None):
        """Split interaction records by ratios.

        Args:
            ratios (list): List of split ratios. No need to be normalized.
            group_by (str, optional): Field name that interaction records should grouped by before splitting.
                Defaults to ``None``

        Returns:
            list: List of :class:`~Dataset`, whose interaction features has been splitted.

        Note:
            Other than the first one, each part is rounded down.
        """
        self.logger.debug('split by ratios [{}], group_by=[{}]'.format(ratios, group_by))
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        if group_by is None:
            tot_cnt = self.__len__()
            split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [range(start, end) for start, end in zip([0] + split_ids, split_ids + [tot_cnt])]
        else:
            grouped_inter_feat_index = self.inter_feat.groupby(by=group_by).groups.values()
            next_index = [[] for i in range(len(ratios))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start: end])

        next_df = [self.inter_feat.loc[index].reset_index(drop=True) for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def _split_index_by_leave_one_out(self, grouped_index, leave_one_num):
        """Split indexes by strategy leave one out.

        Args:
            grouped_index (pandas.DataFrameGroupBy): Index to be splitted.
            leave_one_num (int): Number of parts whose length is expected to be ``1``.

        Returns:
            list: List of index that has been splitted.
        """
        next_index = [[] for i in range(leave_one_num + 1)]
        for index in grouped_index:
            index = list(index)
            tot_cnt = len(index)
            legal_leave_one_num = min(leave_one_num, tot_cnt - 1)
            pr = tot_cnt - legal_leave_one_num
            next_index[0].extend(index[:pr])
            for i in range(legal_leave_one_num):
                next_index[-legal_leave_one_num + i].append(index[pr])
                pr += 1
        return next_index

    def leave_one_out(self, group_by, leave_one_num=1):
        """Split interaction records by leave one out strategy.

        Args:
            group_by (str): Field name that interaction records should grouped by before splitting.
            leave_one_num (int, optional): Number of parts whose length is expected to be ``1``.
                Defaults to ``1``.

        Returns:
            list: List of :class:`~Dataset`, whose interaction features has been splitted.
        """
        self.logger.debug('leave one out, group_by=[{}], leave_one_num=[{}]'.format(group_by, leave_one_num))
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        grouped_inter_feat_index = self.inter_feat.groupby(by=group_by).groups.values()
        next_index = self._split_index_by_leave_one_out(grouped_inter_feat_index, leave_one_num)
        next_df = [self.inter_feat.loc[index].reset_index(drop=True) for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.inter_feat = self.inter_feat.sample(frac=1).reset_index(drop=True)

    def sort(self, by, ascending=True):
        """Sort the interaction records inplace.

        Args:
            by (str): Field that as the key in the sorting process.
            ascending (bool, optional): Results are ascending if ``True``, otherwise descending.
                Defaults to ``True``
        """
        self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True, ignore_index=True)

    def build(self, eval_setting):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of builded :class:`Dataset`.
        """
        ordering_args = eval_setting.ordering_args
        if ordering_args['strategy'] == 'shuffle':
            self.shuffle()
        elif ordering_args['strategy'] == 'by':
            self.sort(by=ordering_args['field'], ascending=ordering_args['ascending'])

        group_field = eval_setting.group_field

        split_args = eval_setting.split_args
        if split_args['strategy'] == 'by_ratio':
            datasets = self.split_by_ratio(split_args['ratios'], group_by=group_field)
        elif split_args['strategy'] == 'by_value':
            raise NotImplementedError()
        elif split_args['strategy'] == 'loo':
            datasets = self.leave_one_out(group_by=group_field, leave_one_num=split_args['leave_one_num'])
        else:
            datasets = self

        return datasets

    def save(self, filepath):
        """Saving this :class:`Dataset` object to local path.

        Args:
            filepath (str): path of saved dir.
        """
        if (filepath is None) or (not os.path.isdir(filepath)):
            raise ValueError('filepath [{}] need to be a dir'.format(filepath))

        self.logger.debug('Saving into [{}]'.format(filepath))
        basic_info = {
            'field2type': self.field2type,
            'field2source': self.field2source,
            'field2id_token': self.field2id_token,
            'field2seqlen': self.field2seqlen
        }

        with open(os.path.join(filepath, 'basic-info.json'), 'w', encoding='utf-8') as file:
            json.dump(basic_info, file)

        feats = ['inter', 'user', 'item']
        for name in feats:
            df = getattr(self, '{}_feat'.format(name))
            if df is not None:
                df.to_csv(os.path.join(filepath, '{}.csv'.format(name)))

    def get_user_feature(self):
        """
        Returns:
            pandas.DataFrame: user features
        """
        if self.user_feat is None:
            self._check_field('uid_field')
            return pd.DataFrame({self.uid_field: np.arange(self.user_num)})
        else:
            return self.user_feat

    def get_item_feature(self):
        """
        Returns:
            pandas.DataFrame: item features
        """
        if self.item_feat is None:
            self._check_field('iid_field')
            return pd.DataFrame({self.iid_field: np.arange(self.item_num)})
        else:
            return self.item_feat

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (pandas.DataFrame): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.num(source_field), self.num(target_field)))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    def _create_graph(self, df_feat, source_field, target_field, form='dgl', value_field=None):
        """Get graph that describe relations between two fields.

        Source and target should be token-like fields.

        For an edge of <src, tgt>, ``graph[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``graph[src, tgt] = df_feat[value_field][src, tgt]``.

        Currently, we support graph in `DGL`_ and `PyG`_.

        Args:
            df_feat (pandas.DataFrame): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Library of graph data structure. Defaults to ``dgl``.
            value_field (str, optional): edge attributes of graph, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            Graph of relations.

        .. _DGL:
            https://www.dgl.ai/

        .. _PyG:
            https://github.com/rusty1s/pytorch_geometric
        """
        tensor_feat = self._dataframe_to_interaction(df_feat)
        src = tensor_feat[source_field]
        tgt = tensor_feat[target_field]

        if form == 'dgl':
            import dgl
            graph = dgl.graph((src, tgt))
            if value_field is not None:
                if isinstance(value_field, str):
                    value_field = {value_field}
                for k in value_field:
                    graph.edata[k] = tensor_feat[k]
            return graph
        elif form == 'pyg':
            from torch_geometric.data import Data
            edge_attr = tensor_feat[value_field] if value_field else None
            graph = Data(edge_index=torch.stack([src, tgt]), edge_attr=edge_attr)
            return graph
        else:
            raise NotImplementedError('graph format [{}] has not been implemented.'.format(form))

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.

        Sparse matrix has shape (user_num, item_num).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.inter_feat, self.uid_field, self.iid_field, form, value_field)

    def _history_matrix(self, row, value_field=None):
        """Get dense matrix describe user/item's history interaction records.

        ``history_matrix[i]`` represents ``i``'s history interacted item_id.

        ``history_value[i]`` represents ``i``'s history interaction records' values.
            ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            row (str): ``user`` or ``item``.
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        self._check_field('uid_field', 'iid_field')

        user_ids, item_ids = self.inter_feat[self.uid_field].values, self.inter_feat[self.iid_field].values
        if value_field is None:
            values = np.ones(len(self.inter_feat))
        else:
            if value_field not in self.inter_feat.columns:
                raise ValueError('value_field [{}] should be one of `inter_feat`\'s features.'.format(value_field))
            values = self.inter_feat[value_field].values

        if row == 'user':
            row_num, max_col_num = self.user_num, self.item_num
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num, max_col_num = self.item_num, self.user_num
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        col_num = np.max(history_len)
        if col_num > max_col_num * 0.2:
            self.logger.warning('max value of {}\'s history interaction records has reached {}% of the total'.format(
                row, col_num / max_col_num * 100,
            ))

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return torch.LongTensor(history_matrix), torch.FloatTensor(history_value), torch.LongTensor(history_len)

    def history_item_matrix(self, value_field=None):
        """Get dense matrix describe user's history interaction records.

        ``history_matrix[i]`` represents user ``i``'s history interacted item_id.

        ``history_value[i]`` represents user ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of user ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        return self._history_matrix(row='user', value_field=value_field)

    def history_user_matrix(self, value_field=None):
        """Get dense matrix describe item's history interaction records.

        ``history_matrix[i]`` represents item ``i``'s history interacted item_id.

        ``history_value[i]`` represents item ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of item ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        return self._history_matrix(row='item', value_field=value_field)

    @dlapi.set()
    def get_preload_weight(self, field):
        """Get preloaded weight matrix, whose rows are sorted by token ids.

        ``0`` is used as padding.

        Args:
            field (str): preloaded feature field name.

        Returns:
            numpy.ndarray: preloaded weight matrix. See :doc:`../user_guide/data/data_args` for details.
        """
        if field not in self._preloaded_weight:
            raise ValueError('field [{}] not in preload_weight'.format(field))
        return self._preloaded_weight[field]

    @dlapi.set()
    def _dataframe_to_interaction(self, data, *args):
        """Convert :class:`pandas.DataFrame` to :class:`~recbole.data.interaction.Interaction`.

        Args:
            data (pandas.DataFrame): data to be converted.

        Returns:
            :class:`~recbole.data.interaction.Interaction`: Converted data.
        """
        data = data.to_dict(orient='list')
        return self._dict_to_interaction(data, *args)

    @dlapi.set()
    def _dict_to_interaction(self, data, *args):
        """Convert :class:`dict` to :class:`~recbole.data.interaction.Interaction`.

        Args:
            data (dict): data to be converted.

        Returns:
            :class:`~recbole.data.interaction.Interaction`: Converted data.
        """
        for k in data:
            ftype = self.field2type[k]
            if ftype == FeatureType.TOKEN:
                data[k] = torch.LongTensor(data[k])
            elif ftype == FeatureType.FLOAT:
                data[k] = torch.FloatTensor(data[k])
            elif ftype == FeatureType.TOKEN_SEQ:
                if isinstance(data[k], np.ndarray):
                    data[k] = torch.LongTensor(data[k][:, :self.field2seqlen[k]])
                else:
                    seq_data = [torch.LongTensor(d[:self.field2seqlen[k]]) for d in data[k]]
                    data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            elif ftype == FeatureType.FLOAT_SEQ:
                if isinstance(data[k], np.ndarray):
                    data[k] = torch.FloatTensor(data[k][:, :self.field2seqlen[k]])
                else:
                    seq_data = [torch.FloatTensor(d[:self.field2seqlen[k]]) for d in data[k]]
                    data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            else:
                raise ValueError('Illegal ftype [{}]'.format(ftype))
        return Interaction(data, *args)
