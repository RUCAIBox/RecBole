# @Time   : 2020/6/28
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/15, 2020/9/3, 2020/9/15
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn

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

from recbox.utils import FeatureSource, FeatureType, ModelType
from recbox.data.interaction import Interaction


class Dataset(object):
    def __init__(self, config, saved_dataset=None):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()

        if saved_dataset is None:
            self._from_scratch(config)
        else:
            self._restore_saved_dataset(saved_dataset)

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

        self._preloaded_weight = {}

        self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.dataset_name, self.dataset_path)
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
        return [feat for feat in [self.inter_feat, self.user_feat, self.item_feat] if feat is not None]

    def _restore_saved_dataset(self, saved_dataset):
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
        self.feat_list = self._build_feat_list()

        self.model_type = self.config['MODEL_TYPE']
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        self.logger.debug('uid_field: {}'.format(self.uid_field))
        self.logger.debug('iid_field: {}'.format(self.iid_field))

    def _load_data(self, token, dataset_path):
        user_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'user'))
        if os.path.isfile(user_feat_path):
            user_feat = self._load_feat(user_feat_path, FeatureSource.USER)
            self.logger.debug('user feature loaded successfully from [{}]'.format(user_feat_path))
        else:
            user_feat = None
            self.logger.debug('[{}] not found, user features are not loaded'.format(user_feat_path))

        item_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'item'))
        if os.path.isfile(item_feat_path):
            item_feat = self._load_feat(item_feat_path, FeatureSource.ITEM)
            self.logger.debug('item feature loaded successfully from [{}]'.format(item_feat_path))
        else:
            item_feat = None
            self.logger.debug('[{}] not found, item features are not loaded'.format(item_feat_path))

        inter_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'inter'))
        if not os.path.isfile(inter_feat_path):
            raise ValueError('File {} not exist'.format(inter_feat_path))

        inter_feat = self._load_feat(inter_feat_path, FeatureSource.INTERACTION)
        self.logger.debug('interaction feature loaded successfully from [{}]'.format(inter_feat_path))

        if user_feat is not None and self.uid_field is None:
            raise ValueError('uid_field must be exist if user_feat exist')

        if item_feat is not None and self.iid_field is None:
            raise ValueError('iid_field must be exist if item_feat exist')

        if self.uid_field in self.field2source:
            self.field2source[self.uid_field] = FeatureSource.USER_ID

        if self.iid_field in self.field2source:
            self.field2source[self.iid_field] = FeatureSource.ITEM_ID

        return inter_feat, user_feat, item_feat

    def _load_feat(self, filepath, source):
        self.logger.debug('loading feature from [{}] (source: [{}])'.format(filepath, source))

        str2ftype = {
            'token': FeatureType.TOKEN,
            'float': FeatureType.FLOAT,
            'token_seq': FeatureType.TOKEN_SEQ,
            'float_seq': FeatureType.FLOAT_SEQ
        }

        if self.config['load_col'] is None:
            load_col = None
        elif source.value not in self.config['load_col']:
            return None
        elif self.config['load_col'][source.value] == '*':
            load_col = None
        else:
            load_col = set(self.config['load_col'][source.value])

        if self.config['unload_col'] is not None and source.value in self.config['unload_col']:
            unload_col = set(self.config['unload_col'][source.value])
        else:
            unload_col = None

        if load_col is not None and unload_col is not None:
            raise ValueError('load_col [{}] and unload_col [{}] can not be setted the same time'.format(
                load_col, unload_col))

        self.logger.debug('\n [{}]:\n\t load_col: [{}]\n\t unload_col: [{}]\n'.format(filepath, load_col, unload_col))

        df = pd.read_csv(filepath, delimiter=self.config['field_separator'])
        field_names = []
        columns = []
        remain_field = set()
        for field_type in df.columns:
            field, ftype = field_type.split(':')
            field_names.append(field)
            if ftype not in str2ftype:
                raise ValueError('Type {} from field {} is not supported'.format(ftype, field))
            ftype = str2ftype[ftype]
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            # TODO user_id & item_id bridge check
            # TODO user_id & item_id not be set in config
            # TODO inter __iter__ loading
            self.field2source[field] = source
            self.field2type[field] = ftype
            if not ftype.value.endswith('seq'):
                self.field2seqlen[field] = 1
            columns.append(field)
            remain_field.add(field)

        if len(columns) == 0:
            self.logger.warning('no columns has been loaded from [{}]'.format(source))
            return None
        df.columns = field_names
        df = df[columns]

        seq_separator = self.config['seq_separator']
        def _token(df, field): pass
        def _float(df, field): pass
        def _token_seq(df, field): df[field] = [_.split(seq_separator) for _ in df[field].values]
        def _float_seq(df, field): df[field] = [list(map(float, _.split(seq_separator))) for _ in df[field].values]
        ftype2func = {
            FeatureType.TOKEN: _token,
            FeatureType.FLOAT: _float,
            FeatureType.TOKEN_SEQ: _token_seq,
            FeatureType.FLOAT_SEQ: _float_seq,
        }
        for field in remain_field:
            ftype = self.field2type[field]
            ftype2func[ftype](df, field)
            if field not in self.field2seqlen:
                self.field2seqlen[field] = max(map(len, df[field].values))
        return df

    def _user_item_feat_preparation(self):
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
            self.feat_list = self._build_feat_list()
            self._fill_nan_flag = True

    def _preload_weight_matrix(self):
        preload_fields = self.config['preload_weight']
        if preload_fields is None:
            return
        drop_flag = self.config['drop_preload_weight']
        if drop_flag is None:
            drop_flag = True
        if not isinstance(preload_fields, list):
            preload_fields = [preload_fields]

        self.logger.debug('preload weight matrix for {}, drop=[{}]'.format(preload_fields, drop_flag))

        feats = [feat for feat in [self.user_feat, self.item_feat] if feat is not None]
        for field in preload_fields:
            used_flag = False
            for feat in feats:
                if field in feat:
                    used_flag = True
                    ftype = self.field2type[field]
                    if ftype == FeatureType.FLOAT:
                        matrix = feat[field].values
                    elif ftype == FeatureType.FLOAT_SEQ:
                        max_len = self.field2seqlen[field]
                        matrix = np.zeros((len(feat[field]), max_len))
                        for i, row in enumerate(feat[field].to_list()):
                            matrix[i] = row[:max_len]
                    else:
                        self.logger.warning('Field [{}] with type [{}] is not \'float\' or \'float_seq\', \
                                             which will not be handled by preload matrix.'.format(field, ftype))
                        continue
                    self._preloaded_weight[field] = matrix
                    if drop_flag:
                        self._del_col(field)
            if not used_flag:
                self.logger.warning('Field [{}] doesn\'t exist, thus not been handled.'.format(field))

    def _fill_nan(self):
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
                    feat[field] = feat[field].apply(lambda x: [0] if (not isinstance(x, np.ndarray) and
                                                                     (not isinstance(x, list)))
                                                                  else x)

    def _normalize(self):
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

    def _filter_by_inter_num(self):
        ban_users = self._get_illegal_ids_by_inter_num(field=self.uid_field,
                                                       max_num=self.config['max_user_inter_num'],
                                                       min_num=self.config['min_user_inter_num'])
        ban_items = self._get_illegal_ids_by_inter_num(field=self.iid_field,
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

    def _get_illegal_ids_by_inter_num(self, field, max_num=None, min_num=None):
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

        self.logger.debug('[{}] illegal_ids_by_inter_num, field=[{}]'.format(len(ids), field))
        return ids

    def _filter_by_field_value(self):
        drop_field = self.config['drop_filter_field']
        changed = False
        changed |= self._drop_by_value(self.config['lowest_val'], lambda x, y: x < y, drop_field)
        changed |= self._drop_by_value(self.config['highest_val'], lambda x, y: x > y, drop_field)
        changed |= self._drop_by_value(self.config['equal_val'], lambda x, y: x != y, drop_field)
        changed |= self._drop_by_value(self.config['not_equal_val'], lambda x, y: x == y, drop_field)

        if not changed:
            return

        if self.user_feat is not None:
            remained_uids = set(self.user_feat[self.uid_field].values)
        elif self.uid_field is not None:
            remained_uids = set(self.inter_feat[self.uid_field].values)

        if self.item_feat is not None:
            remained_iids = set(self.item_feat[self.iid_field].values)
        elif self.iid_field is not None:
            remained_iids = set(self.inter_feat[self.iid_field].values)

        remained_inter = pd.Series(True, index=self.inter_feat.index)
        if self.uid_field is not None:
            remained_inter &= self.inter_feat[self.uid_field].isin(remained_uids)
        if self.iid_field is not None:
            remained_inter &= self.inter_feat[self.iid_field].isin(remained_iids)
        self.logger.debug('[{}] interactions are remained after filtering'.format(len(remained_inter)))
        self.inter_feat.drop(self.inter_feat.index[~remained_inter], inplace=True)

    def _reset_index(self):
        for feat in self.feat_list:
            feat.reset_index(drop=True, inplace=True)

    def _drop_by_value(self, val, cmp, drop_field=False):
        if val is None:
            return False

        self.logger.debug('drop_by_value: val={}, drop=[{}]'.format(val, drop_field))
        for field in val:
            if field not in self.field2type:
                raise ValueError('field [{}] not defined in dataset'.format(field))
            if self.field2type[field] not in {FeatureType.FLOAT, FeatureType.FLOAT_SEQ}:
                raise ValueError('field [{}] is not float like field in dataset, which can\'t be filter'.format(field))
            for feat in self.feat_list:
                if field in feat:
                    feat.drop(feat.index[cmp(feat[field].values, val[field])], inplace=True)
            if drop_field:
                self._del_col(field)
        return True

    def _del_col(self, field):
        self.logger.debug('delete column [{}]'.format(field))
        for feat in self.feat_list:
            if field in feat:
                feat.drop(columns=field, inplace=True)
        for dct in [self.field2id_token, self.field2seqlen, self.field2source, self.field2type]:
            if field in dct:
                del dct[field]

    def _set_label_by_threshold(self):
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
        fields_in_same_space = self.config['fields_in_same_space'] or []
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
                    raise ValueError('field [{}] is not a token like field'.format(field))

        fields_in_same_space.extend(additional)
        return fields_in_same_space

    def _remap_ID_all(self):
        fields_in_same_space = self._get_fields_in_same_space()
        self.logger.debug('fields_in_same_space: {}'.format(fields_in_same_space))
        for field_set in fields_in_same_space:
            remap_list = []
            for field, feat in zip([self.uid_field, self.iid_field], [self.user_feat, self.item_feat]):
                if field in field_set:
                    field_set.remove(field)
                    remap_list.append((self.inter_feat, field, FeatureType.TOKEN))
                    if feat is not None:
                        remap_list.append((feat, field, FeatureType.TOKEN))
            for field in field_set:
                source = self.field2source[field]
                feat = getattr(self, '{}_feat'.format(source.value))
                ftype = self.field2type[field]
                remap_list.append((feat, field, ftype))
            self._remap(remap_list)

    def _remap(self, remap_list, overwrite=True):
        tokens = []
        for feat, field, ftype in remap_list:
            if ftype == FeatureType.TOKEN:
                tokens.append(feat[field].values)
            elif ftype == FeatureType.TOKEN_SEQ:
                tokens.append(feat[field].agg(np.concatenate))
        split_point = np.cumsum(list(map(len, tokens)))[:-1]
        tokens = np.concatenate(tokens)
        new_ids_list, mp = pd.factorize(tokens)
        new_ids_list = np.split(new_ids_list + 1, split_point)
        mp = ['[PAD]'] + list(mp)

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if overwrite or (field not in self.field2id_token):
                self.field2id_token[field] = mp
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    def num(self, field):
        if field not in self.field2type:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        if self.field2type[field] not in {FeatureType.TOKEN, FeatureType.TOKEN_SEQ}:
            return self.field2seqlen[field]
        else:
            return len(self.field2id_token[field])

    def fields(self, ftype=None):
        ftype = set(ftype) if ftype is not None else set(FeatureType)
        ret = []
        for field in self.field2type:
            tp = self.field2type[field]
            if tp in ftype:
                ret.append(field)
        return ret

    @property
    def float_like_fields(self):
        return self.fields([FeatureType.FLOAT, FeatureType.FLOAT_SEQ])

    @property
    def token_like_fields(self):
        return self.fields([FeatureType.TOKEN, FeatureType.TOKEN_SEQ])

    @property
    def seq_fields(self):
        return self.fields([FeatureType.FLOAT_SEQ, FeatureType.TOKEN_SEQ])

    @property
    def non_seq_fields(self):
        return self.fields([FeatureType.FLOAT, FeatureType.TOKEN])

    def set_field_property(self, field, field2type, field2source, field2seqlen):
        self.field2type[field] = field2type
        self.field2source[field] = field2source
        self.field2seqlen[field] = field2seqlen

    def copy_field_property(self, dest_field, source_field):
        self.field2type[dest_field] = self.field2type[source_field]
        self.field2source[dest_field] = self.field2source[source_field]
        self.field2seqlen[dest_field] = self.field2seqlen[source_field]

    @property
    def user_num(self):
        self._check_field('uid_field')
        return self.num(self.uid_field)

    @property
    def item_num(self):
        self._check_field('iid_field')
        return self.num(self.iid_field)

    @property
    def inter_num(self):
        return len(self.inter_feat)

    @property
    def avg_actions_of_users(self):
        return np.mean(self.inter_feat.groupby(self.uid_field).size())

    @property
    def avg_actions_of_items(self):
        return np.mean(self.inter_feat.groupby(self.iid_field).size())

    @property
    def sparsity(self):
        return 1 - self.inter_num / self.user_num / self.item_num

    @property
    def uid2items(self):
        self._check_field('uid_field', 'iid_field')
        uid2items = dict()
        columns = [self.uid_field, self.iid_field]
        for uid, iid in self.inter_feat[columns].values:
            if uid not in uid2items:
                uid2items[uid] = []
            uid2items[uid].append(iid)
        return pd.DataFrame(list(uid2items.items()), columns=columns)

    @property
    def uid2index(self):
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

    def prepare_data_augmentation(self):
        self.logger.debug('prepare_data_augmentation')
        if hasattr(self, 'uid_list'):
            return self.uid_list, self.item_list_index, self.target_index, self.item_list_length

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].values):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_list_length = np.array(item_list_length)
        return self.uid_list, self.item_list_index, self.target_index, self.item_list_length

    def _check_field(self, *field_names):
        for field_name in field_names:
            if getattr(self, field_name, None) is None:
                raise ValueError('{} isn\'t set'.format(field_name))

    def join(self, df):
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
        info = []
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

    # def __iter__(self):
    #     return self

    # TODO next func
    # def next(self):
    #     pass

    # TODO copy
    def copy(self, new_inter_feat):
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def _calcu_split_ids(self, tot, ratios):
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def split_by_ratio(self, ratios, group_by=None):
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
        self.logger.debug('leave one out, group_by=[{}], leave_one_num=[{}]'.format(group_by, leave_one_num))
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        if self.model_type == ModelType.SEQUENTIAL:
            self.prepare_data_augmentation()
            grouped_index = pd.DataFrame(self.uid_list).groupby(by=0).groups.values()
            next_index = self._split_index_by_leave_one_out(grouped_index, leave_one_num)
            next_ds = []
            for index in next_index:
                ds = copy.copy(self)
                for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length']:
                    setattr(ds, field, np.array(getattr(ds, field)[index]))
                next_ds.append(ds)
        else:
            grouped_inter_feat_index = self.inter_feat.groupby(by=group_by).groups.values()
            next_index = self._split_index_by_leave_one_out(grouped_inter_feat_index, leave_one_num)
            next_df = [self.inter_feat.loc[index].reset_index(drop=True) for index in next_index]
            next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def shuffle(self):
        self.inter_feat = self.inter_feat.sample(frac=1).reset_index(drop=True)

    def sort(self, by, ascending=True):
        self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True, ignore_index=True)

    # TODO
    def build(self, eval_setting):
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
        if self.user_feat is None:
            self._check_field('uid_field')
            return pd.DataFrame({self.uid_field: np.arange(self.user_num)})
        else:
            return self.user_feat

    def get_item_feature(self):
        if self.item_feat is None:
            self._check_field('iid_field')
            return pd.DataFrame({self.iid_field: np.arange(self.item_num)})
        else:
            return self.item_feat

    def inter_matrix(self, form='coo', value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')

        uids = self.inter_feat[self.uid_field].values
        iids = self.inter_feat[self.iid_field].values
        if value_field is None:
            data = np.ones(len(self.inter_feat))
        else:
            if value_field not in self.field2source:
                raise ValueError('value_field [{}] not exist.'.format(value_field))
            if self.field2source[value_field] != FeatureSource.INTERACTION:
                raise ValueError('value_field [{}] can only be one of the interaction features'.format(value_field))
            data = self.inter_feat[value_field].values
        mat = coo_matrix((data, (uids, iids)), shape=(self.user_num, self.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('interaction matrix format [{}] has not been implemented.')

    def _history_matrix(self, row):
        self._check_field(self.uid_field, self.iid_field)

        user_ids = self.inter_feat[self.uid_field].values
        item_ids = self.inter_feat[self.iid_field].values

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
        history_len[:] = 0
        for row_id, col_id in zip(row_ids, col_ids):
            history_matrix[history_len[row_id]] = col_id
            history_len[row_id] += 1

        return torch.LongTensor(history_matrix), torch.LongTensor(history_len)

    def history_item_matrix(self):
        return self._history_matrix(row='user')

    def history_user_matrix(self):
        return self._history_matrix(row='item')

    def get_preload_weight(self, field):
        if field not in self._preloaded_weight:
            raise ValueError('field [{}] not in preload_weight'.format(field))
        return self._preloaded_weight[field]

    def _dataframe_to_interaction(self, data, *args):
        data = data.to_dict(orient='list')
        return self._dict_to_interaction(data, *args)

    def _dict_to_interaction(self, data, *args):
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
