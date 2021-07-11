# -*- coding: utf-8 -*-
# @Time   : 2021/1/5
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time    :   2020/1/5, 2021/7/1
# @Author  :   Yushuo Chen, Xingyu Pan
# @email   :   chenyushuo@ruc.edu.cn, xy_pan@foxmail.com

import logging
import os

import pytest

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

current_path = os.path.dirname(os.path.realpath(__file__))


def new_dataloader(config_dict=None, config_file_list=None):
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    return data_preparation(config, dataset)


class TestGeneralDataloader:
    def test_general_dataloader(self):
        train_batch_size = 6
        eval_batch_size = 2
        config_dict = {
            'model': 'BPR',
            'dataset': 'general_dataloader',
            'data_path': current_path,
            'load_col': None,
            'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'TO', 'mode': 'none'},
            'training_neg_sample_num': 0,
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_dataloader(data, item_list, batch_size):
            data.shuffle = False
            pr = 0
            for batch_data in data:
                batch_item_list = item_list[pr: pr + batch_size]
                assert (batch_data['item_id'].numpy() == batch_item_list).all()
                pr += batch_size

        check_dataloader(train_data, list(range(1, 41)), train_batch_size)
        check_dataloader(valid_data, list(range(41, 46)), max(eval_batch_size, 5))
        check_dataloader(test_data, list(range(46, 51)), max(eval_batch_size, 5))

    def test_general_neg_sample_dataloader_in_pair_wise(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            'model': 'BPR',
            'dataset': 'general_dataloader',
            'data_path': current_path,
            'load_col': None,
            'training_neg_sample_num': 1,
            'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'TO', 'mode': 'full'},
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        train_data.shuffle = False
        train_item_list = list(range(1, 41))
        pr = 0
        for batch_data in train_data:
            batch_item_list = train_item_list[pr: pr + train_batch_size]
            assert (batch_data['item_id'].numpy() == batch_item_list).all()
            assert (batch_data['item_id'] == batch_data['price']).all()
            assert (40 < batch_data['neg_item_id']).all()
            assert (batch_data['neg_item_id'] <= 100).all()
            assert (batch_data['neg_item_id'] == batch_data['neg_price']).all()
            pr += train_batch_size

    def test_general_neg_sample_dataloader_in_point_wise(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            'model': 'DMF',
            'dataset': 'general_dataloader',
            'data_path': current_path,
            'load_col': None,
            'training_neg_sample_num': 1,
            'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'TO', 'mode': 'full'},
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        train_data.shuffle = False
        train_item_list = list(range(1, 41))
        pr = 0
        for batch_data in train_data:
            step = len(batch_data) // 2
            batch_item_list = train_item_list[pr: pr + step]
            assert (batch_data['item_id'][: step].numpy() == batch_item_list).all()
            assert (40 < batch_data['item_id'][step:]).all()
            assert (batch_data['item_id'][step:] <= 100).all()
            assert (batch_data['item_id'] == batch_data['price']).all()
            pr += step

    def test_general_full_dataloader(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            'model': 'BPR',
            'dataset': 'general_full_dataloader',
            'data_path': current_path,
            'load_col': None,
            'training_neg_sample_num': 1,
            'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'TO', 'mode': 'full'},
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                user_df, history_index, swap_row, swap_col_after, swap_col_before = batch_data
                history_row, history_col = history_index
                assert len(user_df) == result[i]['len_user_df']
                assert (user_df['user_id'].numpy() == result[i]['user_df_user_id']).all()
                assert (user_df.pos_len_list == result[i]['pos_len_list']).all()
                assert (user_df.user_len_list == result[i]['user_len_list']).all()
                assert len(history_row) == len(history_col) == result[i]['history_len']
                assert (history_row.numpy() == result[i]['history_row']).all()
                assert (history_col.numpy() == result[i]['history_col']).all()
                assert len(swap_row) == len(swap_col_after) == len(swap_col_before) == result[i]['swap_len']
                assert (swap_row.numpy() == result[i]['swap_row']).all()
                assert (swap_col_after.numpy() == result[i]['swap_col_after']).all()
                assert (swap_col_before.numpy() == result[i]['swap_col_before']).all()

        valid_result = [
            {
                'len_user_df': 1,
                'user_df_user_id': [1],
                'pos_len_list': [5],
                'user_len_list': [101],
                'history_len': 40,
                'history_row': 0,
                'history_col': list(range(1, 41)),
                'swap_len': 10,
                'swap_row': 0,
                'swap_col_after': [0, 1, 2, 3, 4, 41, 42, 43, 44, 45],
                'swap_col_before': [45, 44, 43, 42, 41, 4, 3, 2, 1, 0],
            },
            {
                'len_user_df': 1,
                'user_df_user_id': [2],
                'pos_len_list': [5],
                'user_len_list': [101],
                'history_len': 37,
                'history_row': 0,
                'history_col': list(range(1, 38)),
                'swap_len': 10,
                'swap_row': 0,
                'swap_col_after': [0, 1, 2, 3, 4, 38, 39, 40, 41, 42],
                'swap_col_before': [42, 41, 40, 39, 38, 4, 3, 2, 1, 0],
            },
            {
                'len_user_df': 1,
                'user_df_user_id': [3],
                'pos_len_list': [1],
                'user_len_list': [101],
                'history_len': 0,
                'history_row': [],
                'history_col': [],
                'swap_len': 2,
                'swap_row': 0,
                'swap_col_after': [0, 1],
                'swap_col_before': [1, 0],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                'len_user_df': 1,
                'user_df_user_id': [1],
                'pos_len_list': [5],
                'user_len_list': [101],
                'history_len': 45,
                'history_row': 0,
                'history_col': list(range(1, 46)),
                'swap_len': 10,
                'swap_row': 0,
                'swap_col_after': [0, 1, 2, 3, 4, 46, 47, 48, 49, 50],
                'swap_col_before': [50, 49, 48, 47, 46, 4, 3, 2, 1, 0],
            },
            {
                'len_user_df': 1,
                'user_df_user_id': [2],
                'pos_len_list': [5],
                'user_len_list': [101],
                'history_len': 37,
                'history_row': 0,
                'history_col': list(range(1, 36)) + [41, 42],
                'swap_len': 10,
                'swap_row': 0,
                'swap_col_after': [0, 1, 2, 3, 4, 36, 37, 38, 39, 40],
                'swap_col_before': [40, 39, 38, 37, 36, 4, 3, 2, 1, 0],
            },
            {
                'len_user_df': 1,
                'user_df_user_id': [3],
                'pos_len_list': [1],
                'user_len_list': [101],
                'history_len': 0,
                'history_row': [],
                'history_col': [],
                'swap_len': 2,
                'swap_row': 0,
                'swap_col_after': [0, 1],
                'swap_col_before': [1, 0],
            },
        ]
        check_result(test_data, test_result)

    def test_general_uni100_dataloader_with_batch_size_in_101(self):
        train_batch_size = 6
        eval_batch_size = 101
        config_dict = {
            'model': 'BPR',
            'dataset': 'general_uni100_dataloader',
            'data_path': current_path,
            'load_col': None,
            'training_neg_sample_num': 1,
            'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'TO', 'mode': 'uni100'},
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert data.batch_size == 202
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                assert result[i]['item_id_check'](batch_data['item_id'])
                assert batch_data.pos_len_list == result[i]['pos_len_list']
                assert batch_data.user_len_list == result[i]['user_len_list']

        valid_result = [
            {
                'item_id_check': lambda data: data[0] == 9
                                              and (8 < data[1:]).all()
                                              and (data[1:] <= 100).all(),
                'pos_len_list': [1],
                'user_len_list': [101],
            },
            {
                'item_id_check': lambda data: data[0] == 1
                                              and (data[1:] != 1).all(),
                'pos_len_list': [1],
                'user_len_list': [101],
            },
            {
                'item_id_check': lambda data: (data[0: 2].numpy() == [17, 18]).all()
                                              and (16 < data[2:]).all()
                                              and (data[2:] <= 100).all(),
                'pos_len_list': [2],
                'user_len_list': [202],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                'item_id_check': lambda data: data[0] == 10
                                              and (9 < data[1:]).all()
                                              and (data[1:] <= 100).all(),
                'pos_len_list': [1],
                'user_len_list': [101],
            },
            {
                'item_id_check': lambda data: data[0] == 1
                                              and (data[1:] != 1).all(),
                'pos_len_list': [1],
                'user_len_list': [101],
            },
            {
                'item_id_check': lambda data: (data[0: 2].numpy() == [19, 20]).all()
                                              and (18 < data[2:]).all()
                                              and (data[2:] <= 100).all(),
                'pos_len_list': [2],
                'user_len_list': [202],
            },
        ]
        check_result(test_data, test_result)

    def test_general_uni100_dataloader_with_batch_size_in_303(self):
        train_batch_size = 6
        eval_batch_size = 303
        config_dict = {
            'model': 'BPR',
            'dataset': 'general_uni100_dataloader',
            'data_path': current_path,
            'load_col': None,
            'training_neg_sample_num': 1,
            'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'TO', 'mode': 'uni100'},
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert data.batch_size == 303
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                assert result[i]['item_id_check'](batch_data['item_id'])
                assert batch_data.pos_len_list == result[i]['pos_len_list']
                assert batch_data.user_len_list == result[i]['user_len_list']

        valid_result = [
            {
                'item_id_check': lambda data: data[0] == 9
                                              and (8 < data[1: 101]).all()
                                              and (data[1: 101] <= 100).all()
                                              and data[101] == 1
                                              and (data[102:202] != 1).all(),
                'pos_len_list': [1, 1],
                'user_len_list': [101, 101],
            },
            {
                'item_id_check': lambda data: (data[0: 2].numpy() == [17, 18]).all()
                                              and (16 < data[2:]).all()
                                              and (data[2:] <= 100).all(),
                'pos_len_list': [2],
                'user_len_list': [202],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                'item_id_check': lambda data: data[0] == 10
                                              and (9 < data[1:101]).all()
                                              and (data[1:101] <= 100).all()
                                              and data[101] == 1
                                              and (data[102:202] != 1).all(),
                'pos_len_list': [1, 1],
                'user_len_list': [101, 101],
            },
            {
                'item_id_check': lambda data: (data[0: 2].numpy() == [19, 20]).all()
                                              and (18 < data[2:]).all()
                                              and (data[2:] <= 100).all(),
                'pos_len_list': [2],
                'user_len_list': [202],
            },
        ]
        check_result(test_data, test_result)


if __name__ == '__main__':
    pytest.main()
