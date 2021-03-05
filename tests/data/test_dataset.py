# -*- coding: utf-8 -*-
# @Time   : 2021/1/3
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time    :   2020/1/3
# @Author  :   Yushuo Chen
# @email   :   chenyushuo@ruc.edu.cn

import logging
import os

import pytest

from recbole.config import Config, EvalSetting
from recbole.data import create_dataset
from recbole.utils import init_seed

current_path = os.path.dirname(os.path.realpath(__file__))


def new_dataset(config_dict=None, config_file_list=None):
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    return create_dataset(config)


def split_dataset(config_dict=None, config_file_list=None):
    dataset = new_dataset(config_dict=config_dict, config_file_list=config_file_list)
    config = dataset.config
    es_str = [_.strip() for _ in config['eval_setting'].split(',')]
    es = EvalSetting(config)
    es.set_ordering_and_splitting(es_str[0])
    return dataset.build(es)


class TestDataset:
    def test_filter_nan_user_or_item(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_nan_user_or_item',
            'data_path': current_path,
            'load_col': None,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1
        assert len(dataset.user_feat) == 3
        assert len(dataset.item_feat) == 3

    def test_remove_duplication_by_first(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'remove_duplication',
            'data_path': current_path,
            'load_col': None,
            'rm_dup_inter': 'first',
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.inter_feat[dataset.time_field][0] == 0

    def test_remove_duplication_by_last(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'remove_duplication',
            'data_path': current_path,
            'load_col': None,
            'rm_dup_inter': 'last',
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.inter_feat[dataset.time_field][0] == 2

    def test_filter_by_field_value_with_lowest_val(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_field_value',
            'data_path': current_path,
            'load_col': None,
            'lowest_val': {
                'timestamp': 4,
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 6

    def test_filter_by_field_value_with_highest_val(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_field_value',
            'data_path': current_path,
            'load_col': None,
            'highest_val': {
                'timestamp': 4,
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 5

    def test_filter_by_field_value_with_equal_val(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_field_value',
            'data_path': current_path,
            'load_col': None,
            'equal_val': {
                'rating': 0,
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 3

    def test_filter_by_field_value_with_not_equal_val(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_field_value',
            'data_path': current_path,
            'load_col': None,
            'not_equal_val': {
                'rating': 4,
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 9

    def test_filter_by_field_value_in_same_field(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_field_value',
            'data_path': current_path,
            'load_col': None,
            'lowest_val': {
                'timestamp': 3,
            },
            'highest_val': {
                'timestamp': 8,
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 6

    def test_filter_by_field_value_in_different_field(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_field_value',
            'data_path': current_path,
            'load_col': None,
            'lowest_val': {
                'timestamp': 3,
            },
            'highest_val': {
                'timestamp': 8,
            },
            'not_equal_val': {
                'rating': 4,
            }
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 5

    def test_filter_inter_by_user_or_item_is_true(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_inter_by_user_or_item',
            'data_path': current_path,
            'load_col': None,
            'filter_inter_by_user_or_item': True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1

    def test_filter_inter_by_user_or_item_is_false(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_inter_by_user_or_item',
            'data_path': current_path,
            'load_col': None,
            'filter_inter_by_user_or_item': False,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2

    def test_filter_by_inter_num_in_min_user_inter_num(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_inter_num',
            'data_path': current_path,
            'load_col': None,
            'min_user_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 6
        assert dataset.item_num == 7

    def test_filter_by_inter_num_in_min_item_inter_num(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_inter_num',
            'data_path': current_path,
            'load_col': None,
            'min_item_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 7
        assert dataset.item_num == 6

    def test_filter_by_inter_num_in_max_user_inter_num(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_inter_num',
            'data_path': current_path,
            'load_col': None,
            'max_user_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 6
        assert dataset.item_num == 7

    def test_filter_by_inter_num_in_max_item_inter_num(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_inter_num',
            'data_path': current_path,
            'load_col': None,
            'max_item_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 5
        assert dataset.item_num == 5

    def test_filter_by_inter_num_in_min_inter_num(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_inter_num',
            'data_path': current_path,
            'load_col': None,
            'min_user_inter_num': 2,
            'min_item_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 5
        assert dataset.item_num == 5

    def test_filter_by_inter_num_in_complex_way(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_by_inter_num',
            'data_path': current_path,
            'load_col': None,
            'max_user_inter_num': 3,
            'min_user_inter_num': 2,
            'min_item_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_rm_dup_by_first_and_filter_value(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'rm_dup_and_filter_value',
            'data_path': current_path,
            'load_col': None,
            'rm_dup_inter': 'first',
            'highest_val': {
                'rating': 4,
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1

    def test_rm_dup_by_last_and_filter_value(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'rm_dup_and_filter_value',
            'data_path': current_path,
            'load_col': None,
            'rm_dup_inter': 'last',
            'highest_val': {
                'rating': 4,
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2

    def test_rm_dup_and_filter_by_inter_num(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'rm_dup_and_filter_by_inter_num',
            'data_path': current_path,
            'load_col': None,
            'rm_dup_inter': 'first',
            'min_user_inter_num': 2,
            'min_item_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_value_and_filter_inter_by_ui(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_value_and_filter_inter_by_ui',
            'data_path': current_path,
            'load_col': None,
            'highest_val': {
                'age': 2,
            },
            'not_equal_val': {
                'price': 2,
            },
            'filter_inter_by_user_or_item': True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_value_and_inter_num(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_value_and_inter_num',
            'data_path': current_path,
            'load_col': None,
            'highest_val': {
                'rating': 0,
                'age': 0,
                'price': 0,
            },
            'min_user_inter_num': 2,
            'min_item_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_inter_by_ui_and_inter_num(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'filter_inter_by_ui_and_inter_num',
            'data_path': current_path,
            'load_col': None,
            'filter_inter_by_user_or_item': True,
            'min_user_inter_num': 2,
            'min_item_inter_num': 2,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_remap_id(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'remap_id',
            'data_path': current_path,
            'load_col': None,
            'fields_in_same_space': None,
        }
        dataset = new_dataset(config_dict=config_dict)
        user_list = dataset.token2id('user_id', ['ua', 'ub', 'uc', 'ud'])
        item_list = dataset.token2id('item_id', ['ia', 'ib', 'ic', 'id'])
        assert (user_list == [1, 2, 3, 4]).all()
        assert (item_list == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat['user_id'] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat['item_id'] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat['add_user'] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat['add_item'] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat['user_list'][0] == [1, 2]).all()
        assert (dataset.inter_feat['user_list'][1] == []).all()
        assert (dataset.inter_feat['user_list'][2] == [3, 4, 1]).all()
        assert (dataset.inter_feat['user_list'][3] == [5]).all()

    def test_remap_id_with_fields_in_same_space(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'remap_id',
            'data_path': current_path,
            'load_col': None,
            'fields_in_same_space': [
                ['user_id', 'add_user', 'user_list'],
                ['item_id', 'add_item'],
            ],
        }
        dataset = new_dataset(config_dict=config_dict)
        user_list = dataset.token2id('user_id', ['ua', 'ub', 'uc', 'ud', 'ue', 'uf'])
        item_list = dataset.token2id('item_id', ['ia', 'ib', 'ic', 'id', 'ie', 'if'])
        assert (user_list == [1, 2, 3, 4, 5, 6]).all()
        assert (item_list == [1, 2, 3, 4, 5, 6]).all()
        assert (dataset.inter_feat['user_id'] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat['item_id'] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat['add_user'] == [2, 5, 4, 6]).all()
        assert (dataset.inter_feat['add_item'] == [5, 3, 6, 1]).all()
        assert (dataset.inter_feat['user_list'][0] == [3, 5]).all()
        assert (dataset.inter_feat['user_list'][1] == []).all()
        assert (dataset.inter_feat['user_list'][2] == [1, 2, 3]).all()
        assert (dataset.inter_feat['user_list'][3] == [6]).all()

    def test_ui_feat_preparation_and_fill_nan(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'ui_feat_preparation_and_fill_nan',
            'data_path': current_path,
            'load_col': None,
            'filter_inter_by_user_or_item': False,
            'normalize_field': None,
            'normalize_all': None,
        }
        dataset = new_dataset(config_dict=config_dict)
        user_token_list = dataset.id2token('user_id', dataset.user_feat['user_id'])
        item_token_list = dataset.id2token('item_id', dataset.item_feat['item_id'])
        assert (user_token_list == ['[PAD]', 'ua', 'ub', 'uc', 'ud', 'ue']).all()
        assert (item_token_list == ['[PAD]', 'ia', 'ib', 'ic', 'id', 'ie']).all()
        assert dataset.inter_feat['rating'][3] == 1.0
        assert dataset.user_feat['age'][4] == 1.5
        assert dataset.item_feat['price'][4] == 1.5
        assert (dataset.inter_feat['time_list'][0] == [1., 2., 3.]).all()
        assert (dataset.inter_feat['time_list'][1] == [2.]).all()
        assert (dataset.inter_feat['time_list'][2] == []).all()
        assert (dataset.inter_feat['time_list'][3] == [5, 4]).all()
        assert (dataset.user_feat['profile'][0] == []).all()
        assert (dataset.user_feat['profile'][1] == [1, 2, 3]).all()
        assert (dataset.user_feat['profile'][2] == []).all()
        assert (dataset.user_feat['profile'][3] == [3]).all()
        assert (dataset.user_feat['profile'][4] == []).all()
        assert (dataset.user_feat['profile'][5] == [3, 2]).all()

    def test_set_label_by_threshold(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'set_label_by_threshold',
            'data_path': current_path,
            'load_col': None,
            'threshold': {
                'rating': 4,
            },
            'normalize_field': None,
            'normalize_all': None,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat['label'] == [1., 0., 1., 0.]).all()

    def test_normalize_all(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'normalize',
            'data_path': current_path,
            'load_col': None,
            'normalize_all': True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat['rating'] == [0., .25, 1., .75, .5]).all()
        assert (dataset.inter_feat['star'] == [1., .5, 0., .25, 0.75]).all()

    def test_normalize_field(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'normalize',
            'data_path': current_path,
            'load_col': None,
            'normalize_field': ['rating'],
            'normalize_all': False,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat['rating'] == [0., .25, 1., .75, .5]).all()
        assert (dataset.inter_feat['star'] == [4., 2., 0., 1., 3.]).all()

    def test_TO_RS_811(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'build_dataset',
            'data_path': current_path,
            'load_col': None,
            'eval_setting': 'TO_RS',
            'split_ratio': [0.8, 0.1, 0.1],
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (train_dataset.inter_feat['item_id'].numpy() == list(range(1, 17))).all()
        assert (valid_dataset.inter_feat['item_id'].numpy() == list(range(17, 19))).all()
        assert (test_dataset.inter_feat['item_id'].numpy() == list(range(19, 21))).all()

    def test_TO_RS_820(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'build_dataset',
            'data_path': current_path,
            'load_col': None,
            'eval_setting': 'TO_RS',
            'split_ratio': [0.8, 0.2, 0.0],
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (train_dataset.inter_feat['item_id'].numpy() == list(range(1, 17))).all()
        assert (valid_dataset.inter_feat['item_id'].numpy() == list(range(17, 21))).all()
        assert len(test_dataset.inter_feat) == 0

    def test_TO_RS_802(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'build_dataset',
            'data_path': current_path,
            'load_col': None,
            'eval_setting': 'TO_RS',
            'split_ratio': [0.8, 0.0, 0.2],
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (train_dataset.inter_feat['item_id'].numpy() == list(range(1, 17))).all()
        assert len(valid_dataset.inter_feat) == 0
        assert (test_dataset.inter_feat['item_id'].numpy() == list(range(17, 21))).all()

    def test_TO_LS(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'build_dataset',
            'data_path': current_path,
            'load_col': None,
            'eval_setting': 'TO_LS',
            'leave_one_num': 2,
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (train_dataset.inter_feat['item_id'].numpy() == list(range(1, 19))).all()
        assert (valid_dataset.inter_feat['item_id'].numpy() == list(range(19, 20))).all()
        assert (test_dataset.inter_feat['item_id'].numpy() == list(range(20, 21))).all()

    def test_RO_RS_811(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'build_dataset',
            'data_path': current_path,
            'load_col': None,
            'eval_setting': 'RO_RS',
            'split_ratio': [0.8, 0.1, 0.1],
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert len(train_dataset.inter_feat) == 16
        assert len(valid_dataset.inter_feat) == 2
        assert len(test_dataset.inter_feat) == 2

    def test_RO_RS_820(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'build_dataset',
            'data_path': current_path,
            'load_col': None,
            'eval_setting': 'RO_RS',
            'split_ratio': [0.8, 0.2, 0.0],
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert len(train_dataset.inter_feat) == 16
        assert len(valid_dataset.inter_feat) == 4
        assert len(test_dataset.inter_feat) == 0

    def test_RO_RS_802(self):
        config_dict = {
            'model': 'BPR',
            'dataset': 'build_dataset',
            'data_path': current_path,
            'load_col': None,
            'eval_setting': 'RO_RS',
            'split_ratio': [0.8, 0.0, 0.2],
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert len(train_dataset.inter_feat) == 16
        assert len(valid_dataset.inter_feat) == 0
        assert len(test_dataset.inter_feat) == 4


if __name__ == "__main__":
    pytest.main()
