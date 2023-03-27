# -*- coding: utf-8 -*-
# @Time   : 2021/1/5
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time    :   2020/1/5, 2021/7/1, 2021/7/19
# @Author  :   Yushuo Chen, Xingyu Pan, Zhichao Feng
# @email   :   chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, fzcbupt@gmail.com

import logging
import os

import pytest
from recbole.data.dataloader.general_dataloader import (
    NegSampleEvalDataLoader,
    FullSortEvalDataLoader,
)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

current_path = os.path.dirname(os.path.realpath(__file__))


def new_dataloader(config_dict=None, config_file_list=None):
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    return data_preparation(config, dataset)


class TestGeneralDataloader:
    def test_general_dataloader(self):
        train_batch_size = 6
        eval_batch_size = 2
        config_dict = {
            "model": "BPR",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "labeled",
            },
            "train_neg_sample_args": None,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_dataloader(data, item_list, batch_size, train=False):
            data.shuffle = False
            pr = 0
            for batch_data in data:
                batch_item_list = item_list[pr : pr + batch_size]
                if train:
                    user_df = batch_data
                else:
                    user_df = batch_data[0]
                assert (user_df["item_id"].numpy() == batch_item_list).all()
                pr += batch_size

        check_dataloader(train_data, list(range(1, 41)), train_batch_size, True)
        check_dataloader(valid_data, list(range(41, 46)), eval_batch_size)
        check_dataloader(test_data, list(range(46, 51)), eval_batch_size)

    def test_general_neg_sample_dataloader_in_pair_wise(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            "model": "BPR",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "full",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        train_data.shuffle = False
        train_item_list = list(range(1, 41))
        pr = 0
        for batch_data in train_data:
            batch_item_list = train_item_list[pr : pr + train_batch_size]
            assert (batch_data["item_id"].numpy() == batch_item_list).all()
            assert (batch_data["item_id"] == batch_data["price"]).all()
            assert (40 < batch_data["neg_item_id"]).all()
            assert (batch_data["neg_item_id"] <= 100).all()
            assert (batch_data["neg_item_id"] == batch_data["neg_price"]).all()
            pr += train_batch_size

    def test_general_neg_sample_dataloader_in_point_wise(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            "model": "DMF",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "full",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        train_data.shuffle = False
        train_item_list = list(range(1, 41))
        pr = 0
        for batch_data in train_data:
            step = len(batch_data) // 2
            batch_item_list = train_item_list[pr : pr + step]
            assert (batch_data["item_id"][:step].numpy() == batch_item_list).all()
            assert (40 < batch_data["item_id"][step:]).all()
            assert (batch_data["item_id"][step:] <= 100).all()
            assert (batch_data["item_id"] == batch_data["price"]).all()
            pr += step

    def test_general_full_dataloader(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            "model": "BPR",
            "dataset": "general_full_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "full",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                user_df, history_index, positive_u, positive_i = batch_data
                history_row, history_col = history_index
                assert len(user_df) == result[i]["len_user_df"]
                assert (
                    user_df["user_id"].numpy() == result[i]["user_df_user_id"]
                ).all()
                assert len(history_row) == len(history_col) == result[i]["history_len"]
                assert (history_row.numpy() == result[i]["history_row"]).all()
                assert (history_col.numpy() == result[i]["history_col"]).all()
                assert (positive_u.numpy() == result[i]["positive_u"]).all()
                assert (positive_i.numpy() == result[i]["positive_i"]).all()

        valid_result = [
            {
                "len_user_df": 1,
                "user_df_user_id": [1],
                "history_len": 40,
                "history_row": 0,
                "history_col": list(range(1, 41)),
                "positive_u": [0, 0, 0, 0, 0],
                "positive_i": [41, 42, 43, 44, 45],
            },
            {
                "len_user_df": 1,
                "user_df_user_id": [2],
                "history_len": 37,
                "history_row": 0,
                "history_col": list(range(1, 38)),
                "positive_u": [0, 0, 0, 0, 0],
                "positive_i": [38, 39, 40, 41, 42],
            },
            {
                "len_user_df": 1,
                "user_df_user_id": [3],
                "history_len": 0,
                "history_row": [],
                "history_col": [],
                "positive_u": [0],
                "positive_i": [1],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                "len_user_df": 1,
                "user_df_user_id": [1],
                "history_len": 45,
                "history_row": 0,
                "history_col": list(range(1, 46)),
                "positive_u": [0, 0, 0, 0, 0],
                "positive_i": [46, 47, 48, 49, 50],
            },
            {
                "len_user_df": 1,
                "user_df_user_id": [2],
                "history_len": 37,
                "history_row": 0,
                "history_col": list(range(1, 36)) + [41, 42],
                "positive_u": [0, 0, 0, 0, 0],
                "positive_i": [36, 37, 38, 39, 40],
            },
            {
                "len_user_df": 1,
                "user_df_user_id": [3],
                "history_len": 0,
                "history_row": [],
                "history_col": [],
                "positive_u": [0],
                "positive_i": [1],
            },
        ]
        check_result(test_data, test_result)

    def test_general_uni100_dataloader_with_batch_size_in_101(self):
        train_batch_size = 6
        eval_batch_size = 101
        config_dict = {
            "model": "BPR",
            "dataset": "general_uni100_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "uni100",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert data._batch_size == 202
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                user_df, row_idx, positive_u, positive_i = batch_data
                assert result[i]["item_id_check"](user_df["item_id"])
                assert (row_idx.numpy() == result[i]["row_idx"]).all()
                assert (positive_u.numpy() == result[i]["positive_u"]).all()
                assert (positive_i.numpy() == result[i]["positive_i"]).all()

        valid_result = [
            {
                "item_id_check": lambda data: data[0] == 9
                and (8 < data[1:]).all()
                and (data[1:] <= 100).all(),
                "row_idx": [0] * 101,
                "positive_u": [0],
                "positive_i": [9],
            },
            {
                "item_id_check": lambda data: data[0] == 1 and (data[1:] != 1).all(),
                "row_idx": [0] * 101,
                "positive_u": [0],
                "positive_i": [1],
            },
            {
                "item_id_check": lambda data: (data[0:2].numpy() == [17, 18]).all()
                and (16 < data[2:]).all()
                and (data[2:] <= 100).all(),
                "row_idx": [0] * 202,
                "positive_u": [0, 0],
                "positive_i": [17, 18],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                "item_id_check": lambda data: data[0] == 10
                and (9 < data[1:]).all()
                and (data[1:] <= 100).all(),
                "row_idx": [0] * 101,
                "positive_u": [0],
                "positive_i": [10],
            },
            {
                "item_id_check": lambda data: data[0] == 1 and (data[1:] != 1).all(),
                "row_idx": [0] * 101,
                "positive_u": [0],
                "positive_i": [1],
            },
            {
                "item_id_check": lambda data: (data[0:2].numpy() == [19, 20]).all()
                and (18 < data[2:]).all()
                and (data[2:] <= 100).all(),
                "row_idx": [0] * 202,
                "positive_u": [0, 0],
                "positive_i": [19, 20],
            },
        ]
        check_result(test_data, test_result)

    def test_general_uni100_dataloader_with_batch_size_in_303(self):
        train_batch_size = 6
        eval_batch_size = 303
        config_dict = {
            "model": "BPR",
            "dataset": "general_uni100_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "uni100",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert data._batch_size == 303
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                user_df, row_idx, positive_u, positive_i = batch_data
                assert result[i]["item_id_check"](user_df["item_id"])
                assert (row_idx.numpy() == result[i]["row_idx"]).all()
                assert (positive_u.numpy() == result[i]["positive_u"]).all()
                assert (positive_i.numpy() == result[i]["positive_i"]).all()

        valid_result = [
            {
                "item_id_check": lambda data: data[0] == 9
                and (8 < data[1:101]).all()
                and (data[1:101] <= 100).all()
                and data[101] == 1
                and (data[102:202] != 1).all(),
                "row_idx": [0] * 101 + [1] * 101,
                "positive_u": [0, 1],
                "positive_i": [9, 1],
            },
            {
                "item_id_check": lambda data: (data[0:2].numpy() == [17, 18]).all()
                and (16 < data[2:]).all()
                and (data[2:] <= 100).all(),
                "row_idx": [0] * 202,
                "positive_u": [0, 0],
                "positive_i": [17, 18],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                "item_id_check": lambda data: data[0] == 10
                and (9 < data[1:101]).all()
                and (data[1:101] <= 100).all()
                and data[101] == 1
                and (data[102:202] != 1).all(),
                "row_idx": [0] * 101 + [1] * 101,
                "positive_u": [0, 1],
                "positive_i": [10, 1],
            },
            {
                "item_id_check": lambda data: (data[0:2].numpy() == [19, 20]).all()
                and (18 < data[2:]).all()
                and (data[2:] <= 100).all(),
                "row_idx": [0] * 202,
                "positive_u": [0, 0],
                "positive_i": [19, 20],
            },
        ]
        check_result(test_data, test_result)

    def test_general_diff_dataloaders_in_valid_test_phases(self):
        train_batch_size = 6
        eval_batch_size = 2
        config_dict = {
            "model": "BPR",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": {"valid": "uni100", "test": "full"},
            },
            "train_neg_sample_args": None,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        assert isinstance(valid_data, NegSampleEvalDataLoader)
        assert isinstance(test_data, FullSortEvalDataLoader)

    def test_general_diff_eval_neg_sample_args_in_valid_test_phases(self):
        train_batch_size = 6
        eval_batch_size = 2
        config_dict = {
            "model": "BPR",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": {"valid": "uni100", "test": "pop200"},
            },
            "train_neg_sample_args": None,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        assert isinstance(valid_data, NegSampleEvalDataLoader)
        assert isinstance(test_data, NegSampleEvalDataLoader)
        assert valid_data.neg_sample_args["distribution"] == "uniform"
        assert valid_data.neg_sample_args["sample_num"] == 100
        assert test_data.neg_sample_args["distribution"] == "popularity"
        assert test_data.neg_sample_args["sample_num"] == 200


if __name__ == "__main__":
    pytest.main()
