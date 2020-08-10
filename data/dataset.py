# @Time   : 2020/6/28
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/6
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

#UPDATE:
#@TIME   : 2020/8/5
#@Author : Xingyu Pan
#@Email  : panxy@ruc.edu.cn 
             
import os
import json
import copy
import pandas as pd
import numpy as np
from .dataloader import *


class Dataset(object):
    def __init__(self, config, saved_dataset=None):
        self.config = config
        self.dataset_name = config['dataset']
        self.support_types = {'token', 'token_seq', 'float', 'float_seq'}

        if saved_dataset is None:
            self._from_scratch(config)
        else:
            self._restore_saved_dataset(saved_dataset)

    def _from_scratch(self, config):
        self.dataset_path = config['data_path']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        if config['seq_len'] is not None:
            self.field2seqlen = config['seq_len']
        else:
            self.field2seqlen = {}

        self.inter_feat = None
        self.user_feat = None
        self.item_feat = None

        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']

        self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.dataset_name, self.dataset_path)

        self.filter_all(
            max_user_inter_count = config['max_user_inter_count'],
            min_user_inter_count = config['min_user_inter_count'],
            max_item_inter_count = config['max_item_inter_count'],
            min_item_inter_count = config['min_item_inter_count'],
            lowest_val=config['lowest_val'],
            highest_val=config['highest_val'],
            equal_val=config['equal_val'],
            not_equal_val=config['not_equal_val'],
            drop=config['drop_filter_field']
        )
        self._remap_ID_all()

    def filter_all(self, max_user_inter_count = None, min_user_inter_count = None,max_item_inter_count = None, min_item_inter_count = None,
                   lowest_val = None, highest_val = None, equal_val = None, not_equal_val = None, drop = False):
        self.filter_users_base_inter(max_count = max_user_inter_count, min_count = min_user_inter_count)
        self.filter_items_base_inter(max_count = max_item_inter_count, min_count = min_item_inter_count)
        if self.item_feat is not None:
            self.filter_items(lowest_val, highest_val, equal_val, not_equal_val, drop)
        if self.user_feat is not None:
            self.filter_users(lowest_val, highest_val, equal_val, not_equal_val, drop)
        self.filter_inters(lowest_val, highest_val, equal_val, not_equal_val, drop)

    def filter_users_base_inter(self, max_count = None,  min_count = None):
        if max_count is not None or min_count is not None:
            temp_dict = {}
            ban_list = []
            ban_user = []
            for user in self.inter_feat["user_id"]:
                    if user in temp_dict:
                        temp_dict[user] += 1
                    else:
                        temp_dict[user] = 1

            if min_count is None:
                for user in self.inter_feat['user_id']:
                    if temp_dict[user] > max_count:
                        ban_list.append(False)
                        ban_user.append(user)
                    else:
                        ban_list.append(True)
            elif max_count is None:
                for user in self.inter_feat['user_id']:
                    if temp_dict[user] < min_count:
                        ban_list.append(False)
                        ban_user.append(user)
                    else:
                        ban_list.append(True)
            else:
                for user in self.inter_feat['user_id']:
                    if temp_dict[user] < min_count or temp_dict[user] > max_count:
                        ban_list.append(False)
                        ban_user.append(user)
                    else:
                        ban_list.append(True)
            self.inter_feat = self.inter_feat[ban_list]
            self.inter_feat.reset_index(drop=True, inplace=True)    
            if self.user_feat is not None:
                ban_list_user = []
                for user in self.user_feat['user_id']:
                    if user in ban_user:
                        ban_list_user.append(False)
                    else:
                        ban_list_user.append(True)
                self.user_feat = self.user_feat[ban_list_user]
                self.user_feat.reset_index(drop=True, inplace=True)    

    def filter_items_base_inter(self, max_count = None,  min_count = None):
        if max_count is not None or min_count is not None:
            temp_dict = {}
            ban_list = []
            ban_item = []
            for item in self.inter_feat["item_id"]:
                    if item in temp_dict:
                        temp_dict[item] += 1
                    else:
                        temp_dict[item] = 1

            if min_count is None:
                for item in self.inter_feat['item_id']:
                    if temp_dict[item] > max_count:
                        ban_list.append(False)
                        ban_item.append(item)
                    else:
                        ban_list.append(True)
            elif max_count is None:
                for item in self.inter_feat['item_id']:
                    if temp_dict[item] < min_count:
                        ban_list.append(False)
                        ban_item.append(item)
                    else:
                        ban_list.append(True)
            else:
                for item in self.inter_feat['item_id']:
                    if temp_dict[item] < min_count or temp_dict[item] > max_count:
                        ban_list.append(False)
                        ban_item.append(item)
                    else:
                        ban_list.append(True)
            self.inter_feat = self.inter_feat[ban_list]
            self.inter_feat.reset_index(drop=True, inplace=True)    
            if self.item_feat is not None:
                ban_list_item = []
                for item in self.item_feat['item_id']:
                    if item in ban_item:
                        ban_list_item.append(False)
                    else:
                        ban_list_item.append(True)
                self.item_feat = self.item_feat[ban_list_item]
                self.item_feat.reset_index(drop=True, inplace=True)    


    def _restore_saved_dataset(self, saved_dataset):
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

        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']

    def _load_data(self, token, dataset_path):
        user_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'user'))
        if os.path.isfile(user_feat_path):
            user_feat = self._load_feat(user_feat_path, 'user')
        else:
            # TODO logging user feat not exist
            user_feat = None

        item_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'item'))
        if os.path.isfile(item_feat_path):
            item_feat = self._load_feat(item_feat_path, 'item')
        else:
            # TODO logging item feat not exist
            item_feat = None

        inter_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'inter'))
        if not os.path.isfile(inter_feat_path):
            raise ValueError('File {} not exist'.format(inter_feat_path))

        inter_feat = self._load_feat(inter_feat_path, 'inter')

        if self.uid_field not in self.field2source:
            raise ValueError('user id field [{}] not exist in [{}]'.format(self.uid_field, self.dataset_name))
        else:
            self.field2source[self.uid_field] = 'user_id'

        if self.iid_field not in self.field2source:
            raise ValueError('item id field [{}] not exist in [{}]'.format(self.iid_field, self.dataset_name))
        else:
            self.field2source[self.iid_field] = 'item_id'

        return inter_feat, user_feat, item_feat

    def _load_feat(self, filepath, source):
        if self.config['load_col'] is None:
            load_col = None
        elif source not in self.config['load_col']:
            return None
        else:
            load_col = set(self.config['load_col'][source])
            if source in {'inter', 'user'}: load_col.add(self.uid_field)
            if source in {'inter', 'item'}: load_col.add(self.iid_field)

        if self.config['unload_col'] is not None and source in self.config['unload_col']:
            unload_col = set(self.config['unload_col'][source])
        else:
            unload_col = None

        if load_col is not None and unload_col is not None:
            raise ValueError('load_col [{}] and unload_col [{}] can not be setted the same time'.format(
                load_col, unload_col))

        df = pd.read_csv(filepath, delimiter=self.config['field_separator'])
        field_names = []
        columns = []
        remain_field = set()
        for field_type in df.columns:
            field, ftype = field_type.split(':')
            field_names.append(field)
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            # TODO user_id & item_id bridge check
            # TODO user_id & item_id not be set in config
            # TODO inter __iter__ loading
            if ftype not in self.support_types:
                raise ValueError('Type {} from field {} is not supported'.format(ftype, field))
            self.field2source[field] = source
            self.field2type[field] = ftype
            if not ftype.endswith('seq'):
                self.field2seqlen[field] = 1
            columns.append(field)
            remain_field.add(field)

        if len(columns) == 0:
            print('source', source)
            return None
        df.columns = field_names
        df = df[columns]

        # TODO  fill nan in df

        seq_separator = self.config['seq_separator']
        def _token(df, field): pass
        def _float(df, field): pass
        def _token_seq(df, field): df[field] = [_.split(seq_separator) for _ in df[field].values]
        def _float_seq(df, field): df[field] = [list(map(float, _.split(seq_separator))) for _ in df[field].values]
        ftype2func = {
            'token': _token,
            'float': _float,
            'token_seq': _token_seq,
            'float_seq': _float_seq,
        }
        for field in remain_field:
            ftype = self.field2type[field]
            ftype2func[ftype](df, field)
            if field not in self.field2seqlen:
                self.field2seqlen[field] = max(map(len, df[field].values))
        return df

    # TODO
    
    def filter_users(self, lowest_val=None, highest_val=None, equal_val=None, not_equal_val=None, drop=False):
        self._filter_users(lowest_val, lambda x, y: x >= y, drop)
        self._filter_users(highest_val, lambda x, y: x <= y, drop)
        self._filter_users(equal_val, lambda x, y: x == y, drop)
        self._filter_users(not_equal_val, lambda x, y: x != y, drop)
        self.user_feat.reset_index(drop=True, inplace=True)

    def _filter_users(self, val, cmp, drop=False):
        if val is not None:
            if 'user' in val:
                val=val['user']
                for field in val:
                    if field not in self.field2type:
                        raise ValueError('field [{}] not defined in dataset'.format(field))
                    self.user_feat = self.user_feat[cmp(self.user_feat[field].values, val[field])]
                    if drop:
                        self._del_col(field)

    def filter_items(self, lowest_val=None, highest_val=None, equal_val=None, not_equal_val=None, drop=False):
        self._filter_items(lowest_val, lambda x, y: x >= y, drop)
        self._filter_items(highest_val, lambda x, y: x <= y, drop)
        self._filter_items(equal_val, lambda x, y: x == y, drop)
        self._filter_items(not_equal_val, lambda x, y: x != y, drop)
        self.item_feat.reset_index(drop=True, inplace=True)

    def _filter_items(self, val, cmp, drop=False):
        if val is not None:
            if 'item' in val:
                val = val['item']
                for field in val:
                    if field not in self.field2type:
                        raise ValueError('field [{}] not defined in dataset'.format(field))
                    self.item_feat = self.item_feat[cmp(self.item_feat[field].values, val[field])]
                    if drop:
                        self._del_col(field)

    def _filter_inters(self, val, cmp, drop=False):
        if val is not None:
            if 'inter' in val:
                val = val['inter']
                for field in val:
                    if field not in self.field2type:
                        raise ValueError('field [{}] not defined in dataset'.format(field))
                    self.inter_feat = self.inter_feat[cmp(self.inter_feat[field].values, val[field])]
                    if drop:
                        self._del_col(field)
    

    def _del_col(self, field):
        for feat in [self.inter_feat, self.user_feat, self.item_feat]:
            if feat is not None and field in feat:
                feat.drop(columns=field, inplace=True)
        for dct in [self.field2id_token, self.field2seqlen, self.field2source, self.field2type]:
            if field in dct:
                del dct[field]

    # TODO
    def filter_inters(self, lowest_val=None, highest_val=None, equal_val=None, not_equal_val=None, drop=False):
        self._filter_inters(lowest_val, lambda x, y: x >= y, drop)
        self._filter_inters(highest_val, lambda x, y: x <= y, drop)
        self._filter_inters(equal_val, lambda x, y: x == y, drop)
        self._filter_inters(not_equal_val, lambda x, y: x != y, drop)

        
        if self.user_feat is not None:
            ban_inter_list=[]
            for user in self.inter_feat['user_id']:
                if user in self.user_feat['user_id']:
                    ban_inter_list.append(True)
                else:
                    ban_inter_list.append(False)
            self.inter_feat = self.inter_feat[ban_inter_list]
        if self.item_feat is not None:
            ban_inter_list=[]
            for user in self.inter_feat['user_id']:
                if user in self.user_feat['user_id']:
                    ban_inter_list.append(True)
                else:
                    ban_inter_list.append(False)
            self.inter_feat = self.inter_feat[ban_inter_list]
        self.inter_feat.reset_index(drop=True, inplace=True)

    def _remap_ID_all(self):
        for field in self.field2type:
            ftype = self.field2type[field]
            fsource = self.field2source[field]
            if ftype == 'token':
                self._remap_ID(fsource, field)
            elif ftype == 'token_seq':
                self._remap_ID_seq(fsource, field)

    def _remap_ID(self, source, field):
        feat_name = '{}_feat'.format(source.split('_')[0])
        feat = getattr(self, feat_name)
        if feat is None:
            feat = pd.DataFrame(columns=[field])
        if source in ['user_id', 'item_id']:
            df = pd.concat([self.inter_feat[field], feat[field]])
            new_ids, mp = pd.factorize(df)
            split_point = [len(self.inter_feat[field])]
            self.inter_feat[field], feat[field] = np.split(new_ids, split_point)
            self.field2id_token[field] = list(mp)
        elif source in ['inter', 'user', 'item']:
            new_ids, mp = pd.factorize(feat[field])
            feat[field] = new_ids
            self.field2id_token[field] = list(mp)

    def _remap_ID_seq(self, source, field):
        if source in ['inter', 'user', 'item']:
            feat_name = '{}_feat'.format(source)
            df = getattr(self, feat_name)
            split_point = np.cumsum(df[field].agg(len))[:-1]
            new_ids, mp = pd.factorize(df[field].agg(np.concatenate))
            new_ids = np.split(new_ids + 1, split_point)
            df[field] = new_ids
            self.field2id_token[field] = [None] + list(mp)

    def num(self, field):
        if field not in self.field2type:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        if self.field2type[field] not in {'token', 'token_seq'}:
            return self.field2seqlen[field]
        else:
            return len(self.field2id_token[field])

    def fields(self, ftype=None):
        ftype = set(ftype) if ftype is not None else {'token', 'token_seq', 'float', 'float_seq'}
        ret = []
        for field in self.field2type:
            tp = self.field2type[field]
            if tp in ftype:
                ret.append(field)
        return ret

    @property
    def user_num(self):
        return self.num(self.uid_field)

    @property
    def item_num(self):
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
        uid2items = dict()
        columns = [self.uid_field, self.iid_field]
        for uid, iid in self.inter_feat[columns].values:
            if uid not in uid2items:
                uid2items[uid] = []
            uid2items[uid].append(iid)
        return pd.DataFrame(list(uid2items.items()), columns=columns)

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
        info = ['The number of users: {}'.format(self.user_num),
                'The number of items: {}'.format(self.item_num),
                'The number of inters: {}'.format(self.inter_num),
                'Average actions of users: {}'.format(self.avg_actions_of_users),
                'Average actions of items: {}'.format(self.avg_actions_of_items),
                'The sparsity of the dataset: {}%'.format(self.sparsity * 100),
                'Remain Fields: {}'.format(list(self.field2type))
                ]
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

    def leave_one_out(self, group_by, leave_one_num=1):
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        grouped_inter_feat_index = self.inter_feat.groupby(by=group_by).groups.values()
        next_index = [[] for i in range(leave_one_num + 1)]
        for grouped_index in grouped_inter_feat_index:
            grouped_index = list(grouped_index)
            tot_cnt = len(grouped_index)
            legal_leave_one_num = min(leave_one_num, tot_cnt - 1)
            pr = tot_cnt - legal_leave_one_num
            next_index[0].extend(grouped_index[:pr])
            for i in range(legal_leave_one_num):
                next_index[-legal_leave_one_num + i].append(grouped_index[pr])
                pr += 1

        next_df = [self.inter_feat.loc[index].reset_index(drop=True) for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def shuffle(self):
        self.inter_feat = self.inter_feat.sample(frac=1).reset_index(drop=True)

    def sort(self, by, ascending):
        self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)

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

    def get_item_feature(self):
        if self.item_feat is None:
            tot_item_cnt = self.num(self.iid_field)
            return pd.DataFrame({self.iid_field: np.arange(tot_item_cnt)})
        else:
            return self.item_feat
