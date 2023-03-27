# -*- coding: utf-8 -*-
# @Time   : 2021/1/3
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time    :   2020/1/3, 2021/7/1, 2021/7/11, 2022/7/10
# @Author  :   Yushuo Chen, Xingyu Pan, Yupeng Hou, Lanling Xu
# @email   :   chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, houyupeng@ruc.edu.cn, xulanling_sherry@163.com

import logging
import os

import pytest

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed

current_path = os.path.dirname(os.path.realpath(__file__))


def new_dataset(config_dict=None, config_file_list=None):
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logging.basicConfig(level=logging.ERROR)
    return create_dataset(config)


def split_dataset(config_dict=None, config_file_list=None):
    dataset = new_dataset(config_dict=config_dict, config_file_list=config_file_list)
    return dataset.build()


class TestDataset:
    def test_filter_nan_user_or_item(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_nan_user_or_item",
            "data_path": current_path,
            "load_col": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1
        assert len(dataset.user_feat) == 3
        assert len(dataset.item_feat) == 3

    def test_remove_duplication_by_first(self):
        config_dict = {
            "model": "BPR",
            "dataset": "remove_duplication",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "first",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.inter_feat[dataset.time_field][0] == 0

    def test_remove_duplication_by_last(self):
        config_dict = {
            "model": "BPR",
            "dataset": "remove_duplication",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "last",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.inter_feat[dataset.time_field][0] == 2

    def test_filter_by_field_value_with_lowest_val(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "timestamp": "[4,inf)",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 6

    def test_filter_by_field_value_with_highest_val(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "timestamp": "(-inf,4]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 5

    def test_filter_by_field_value_with_equal_val(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "rating": "[0,0]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 3

    def test_filter_by_field_value_with_not_equal_val(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "rating": "(-inf,4);(4,inf)",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 9

    def test_filter_by_field_value_in_same_field(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "timestamp": "[3,8]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 6

    def test_filter_by_field_value_in_different_field(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "timestamp": "[3,8]",
                "rating": "(-inf,4);(4,inf)",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 5

    def test_filter_inter_by_user_or_item_is_true(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_inter_by_user_or_item",
            "data_path": current_path,
            "load_col": None,
            "filter_inter_by_user_or_item": True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1

    def test_filter_inter_by_user_or_item_is_false(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_inter_by_user_or_item",
            "data_path": current_path,
            "load_col": None,
            "filter_inter_by_user_or_item": False,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2

    def test_filter_by_inter_num_in_min_user_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "user_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 6
        assert dataset.item_num == 7

    def test_filter_by_inter_num_in_min_item_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 7
        assert dataset.item_num == 6

    def test_filter_by_inter_num_in_max_user_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "user_inter_num_interval": "(-inf,2]",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 6
        assert dataset.item_num == 7

    def test_filter_by_inter_num_in_max_item_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "item_inter_num_interval": "(-inf,2]",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 5
        assert dataset.item_num == 5

    def test_filter_by_inter_num_in_min_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "user_inter_num_interval": "[2,inf)",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 5
        assert dataset.item_num == 5

    def test_filter_by_inter_num_in_complex_way(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "user_inter_num_interval": "[2,3]",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_rm_dup_by_first_and_filter_value(self):
        config_dict = {
            "model": "BPR",
            "dataset": "rm_dup_and_filter_value",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "first",
            "val_interval": {
                "rating": "(-inf,4]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1

    def test_rm_dup_by_last_and_filter_value(self):
        config_dict = {
            "model": "BPR",
            "dataset": "rm_dup_and_filter_value",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "last",
            "val_interval": {
                "rating": "(-inf,4]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2

    def test_rm_dup_and_filter_by_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "rm_dup_and_filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "first",
            "user_inter_num_interval": "[2,inf)",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_value_and_filter_inter_by_ui(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_value_and_filter_inter_by_ui",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "age": "(-inf,2]",
                "price": "(-inf,2);(2,inf)",
            },
            "filter_inter_by_user_or_item": True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_value_and_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_value_and_inter_num",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "rating": "(-inf,0]",
                "age": "(-inf,0]",
                "price": "(-inf,0]",
            },
            "user_inter_num_interval": "[2,inf)",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_inter_by_ui_and_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_inter_by_ui_and_inter_num",
            "data_path": current_path,
            "load_col": None,
            "filter_inter_by_user_or_item": True,
            "user_inter_num_interval": "[2,inf)",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_remap_id(self):
        config_dict = {
            "model": "BPR",
            "dataset": "remap_id",
            "data_path": current_path,
            "load_col": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        user_list = dataset.token2id("user_id", ["ua", "ub", "uc", "ud"])
        item_list = dataset.token2id("item_id", ["ia", "ib", "ic", "id"])
        assert (user_list == [1, 2, 3, 4]).all()
        assert (item_list == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["user_id"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["item_id"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["add_user"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["add_item"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["user_list"][0] == [1, 2]).all()
        assert (dataset.inter_feat["user_list"][1] == []).all()
        assert (dataset.inter_feat["user_list"][2] == [3, 4, 1]).all()
        assert (dataset.inter_feat["user_list"][3] == [5]).all()

    def test_remap_id_with_alias(self):
        config_dict = {
            "model": "BPR",
            "dataset": "remap_id",
            "data_path": current_path,
            "load_col": None,
            "alias_of_user_id": ["add_user", "user_list"],
            "alias_of_item_id": ["add_item"],
        }
        dataset = new_dataset(config_dict=config_dict)
        user_list = dataset.token2id("user_id", ["ua", "ub", "uc", "ud", "ue", "uf"])
        item_list = dataset.token2id("item_id", ["ia", "ib", "ic", "id", "ie", "if"])
        assert (user_list == [1, 2, 3, 4, 5, 6]).all()
        assert (item_list == [1, 2, 3, 4, 5, 6]).all()
        assert (dataset.inter_feat["user_id"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["item_id"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["add_user"] == [2, 5, 4, 6]).all()
        assert (dataset.inter_feat["add_item"] == [5, 3, 6, 1]).all()
        assert (dataset.inter_feat["user_list"][0] == [3, 5]).all()
        assert (dataset.inter_feat["user_list"][1] == []).all()
        assert (dataset.inter_feat["user_list"][2] == [1, 2, 3]).all()
        assert (dataset.inter_feat["user_list"][3] == [6]).all()

    def test_ui_feat_preparation_and_fill_nan(self):
        config_dict = {
            "model": "BPR",
            "dataset": "ui_feat_preparation_and_fill_nan",
            "data_path": current_path,
            "load_col": None,
            "filter_inter_by_user_or_item": False,
            "normalize_field": None,
            "normalize_all": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        user_token_list = dataset.id2token("user_id", dataset.user_feat["user_id"])
        item_token_list = dataset.id2token("item_id", dataset.item_feat["item_id"])
        assert (user_token_list == ["[PAD]", "ua", "ub", "uc", "ud", "ue"]).all()
        assert (item_token_list == ["[PAD]", "ia", "ib", "ic", "id", "ie"]).all()
        assert dataset.inter_feat["rating"][3] == 1.0
        assert dataset.user_feat["age"][4] == 1.5
        assert dataset.item_feat["price"][4] == 1.5
        assert (dataset.inter_feat["time_list"][0] == [1.0, 2.0, 3.0]).all()
        assert (dataset.inter_feat["time_list"][1] == [2.0]).all()
        assert (dataset.inter_feat["time_list"][2] == []).all()
        assert (dataset.inter_feat["time_list"][3] == [5, 4]).all()
        assert (dataset.user_feat["profile"][0] == []).all()
        assert (dataset.user_feat["profile"][1] == [1, 2, 3]).all()
        assert (dataset.user_feat["profile"][2] == []).all()
        assert (dataset.user_feat["profile"][3] == [3]).all()
        assert (dataset.user_feat["profile"][4] == []).all()
        assert (dataset.user_feat["profile"][5] == [3, 2]).all()

    def test_set_label_by_threshold(self):
        config_dict = {
            "model": "BPR",
            "dataset": "set_label_by_threshold",
            "data_path": current_path,
            "load_col": None,
            "threshold": {
                "rating": 4,
            },
            "normalize_field": None,
            "normalize_all": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat["label"] == [1.0, 0.0, 1.0, 0.0]).all()

    def test_normalize_all(self):
        config_dict = {
            "model": "BPR",
            "dataset": "normalize",
            "data_path": current_path,
            "load_col": None,
            "normalize_all": True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat["rating"] == [0.0, 0.25, 1.0, 0.75, 0.5]).all()
        assert (dataset.inter_feat["star"] == [1.0, 0.5, 0.0, 0.25, 0.75]).all()

    def test_normalize_field(self):
        config_dict = {
            "model": "BPR",
            "dataset": "normalize",
            "data_path": current_path,
            "load_col": None,
            "normalize_field": ["rating"],
            "normalize_all": False,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat["rating"] == [0.0, 0.25, 1.0, 0.75, 0.5]).all()
        assert (dataset.inter_feat["star"] == [4.0, 2.0, 0.0, 1.0, 3.0]).all()

    def test_TO_RS_811(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 17))
            + [1]
            + [1]
            + [1]
            + [1, 2, 3]
            + list(range(1, 8))
            + list(range(1, 9))
            + list(range(1, 10))
        ).all()
        assert (
            valid_dataset.inter_feat["item_id"].numpy()
            == list(range(17, 19)) + [] + [] + [2] + [4] + [8] + [9] + [10]
        ).all()
        assert (
            test_dataset.inter_feat["item_id"].numpy()
            == list(range(19, 21)) + [] + [2] + [3] + [5] + [9] + [10] + [11]
        ).all()

    def test_TO_RS_820(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.2, 0.0]},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 17))
            + [1]
            + [1]
            + [1, 2]
            + [1, 2, 3, 4]
            + list(range(1, 9))
            + list(range(1, 9))
            + list(range(1, 10))
        ).all()
        assert (
            valid_dataset.inter_feat["item_id"].numpy()
            == list(range(17, 21)) + [] + [2] + [3] + [5] + [9] + [9, 10] + [10, 11]
        ).all()
        assert len(test_dataset.inter_feat) == 0

    def test_TO_RS_802(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.0, 0.2]},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 17))
            + [1]
            + [1]
            + [1, 2]
            + [1, 2, 3, 4]
            + list(range(1, 9))
            + list(range(1, 9))
            + list(range(1, 10))
        ).all()
        assert len(valid_dataset.inter_feat) == 0
        assert (
            test_dataset.inter_feat["item_id"].numpy()
            == list(range(17, 21)) + [] + [2] + [3] + [5] + [9] + [9, 10] + [10, 11]
        ).all()

    def test_TO_LS_valid_and_test(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 19))
            + [1]
            + [1]
            + [1]
            + [1, 2, 3]
            + list(range(1, 8))
            + list(range(1, 9))
            + list(range(1, 10))
        ).all()
        assert (
            valid_dataset.inter_feat["item_id"].numpy()
            == list(range(19, 20)) + [] + [] + [2] + [4] + [8] + [9] + [10]
        ).all()
        assert (
            test_dataset.inter_feat["item_id"].numpy()
            == list(range(20, 21)) + [] + [2] + [3] + [5] + [9] + [10] + [11]
        ).all()

    def test_TO_LS_valid_only(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"LS": "valid_only"},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 20))
            + [1]
            + [1]
            + [1, 2]
            + [1, 2, 3, 4]
            + list(range(1, 9))
            + list(range(1, 10))
            + list(range(1, 11))
        ).all()
        assert (
            valid_dataset.inter_feat["item_id"].numpy()
            == list(range(20, 21)) + [] + [2] + [3] + [5] + [9] + [10] + [11]
        ).all()
        assert len(test_dataset.inter_feat) == 0

    def test_TO_LS_test_only(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"LS": "test_only"},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 20))
            + [1]
            + [1]
            + [1, 2]
            + [1, 2, 3, 4]
            + list(range(1, 9))
            + list(range(1, 10))
            + list(range(1, 11))
        ).all()
        assert len(valid_dataset.inter_feat) == 0
        assert (
            test_dataset.inter_feat["item_id"].numpy()
            == list(range(20, 21)) + [] + [2] + [3] + [5] + [9] + [10] + [11]
        ).all()

    def test_RO_RS_811(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "RO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert len(train_dataset.inter_feat) == 16 + 1 + 1 + 1 + 3 + 7 + 8 + 9
        assert len(valid_dataset.inter_feat) == 2 + 0 + 0 + 1 + 1 + 1 + 1 + 1
        assert len(test_dataset.inter_feat) == 2 + 0 + 1 + 1 + 1 + 1 + 1 + 1

    def test_RO_RS_820(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.2, 0.0]},
                "order": "RO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert len(train_dataset.inter_feat) == 16 + 1 + 1 + 2 + 4 + 8 + 8 + 9
        assert len(valid_dataset.inter_feat) == 4 + 0 + 1 + 1 + 1 + 1 + 2 + 2
        assert len(test_dataset.inter_feat) == 0

    def test_RO_RS_802(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.0, 0.2]},
                "order": "RO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert len(train_dataset.inter_feat) == 16 + 1 + 1 + 2 + 4 + 8 + 8 + 9
        assert len(valid_dataset.inter_feat) == 0
        assert len(test_dataset.inter_feat) == 4 + 0 + 1 + 1 + 1 + 1 + 2 + 2


class TestSeqDataset:
    def test_seq_leave_one_out(self):
        config_dict = {
            "model": "GRU4Rec",
            "dataset": "seq_dataset",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat[train_dataset.uid_field].numpy()
            == [1, 1, 1, 1, 1, 4, 2, 2, 3]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_id_list_field][:, :5].numpy()
            == [
                [1, 0, 0, 0, 0],
                [1, 2, 0, 0, 0],
                [1, 2, 3, 0, 0],
                [1, 2, 3, 4, 0],
                [1, 2, 3, 4, 5],
                [3, 0, 0, 0, 0],
                [4, 0, 0, 0, 0],
                [4, 5, 0, 0, 0],
                [4, 0, 0, 0, 0],
            ]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.iid_field].numpy()
            == [2, 3, 4, 5, 6, 4, 5, 6, 5]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_list_length_field].numpy()
            == [1, 2, 3, 4, 5, 1, 1, 2, 1]
        ).all()

        assert (
            valid_dataset.inter_feat[valid_dataset.uid_field].numpy() == [1, 2]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_id_list_field][:, :6].numpy()
            == [[1, 2, 3, 4, 5, 6], [4, 5, 6, 0, 0, 0]]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.iid_field].numpy() == [7, 7]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_list_length_field].numpy()
            == [6, 3]
        ).all()

        assert (
            test_dataset.inter_feat[test_dataset.uid_field].numpy() == [1, 2, 3]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_id_list_field][:, :7].numpy()
            == [[1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0]]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.iid_field].numpy() == [8, 8, 6]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_list_length_field].numpy()
            == [7, 4, 2]
        ).all()

        assert (
            train_dataset.inter_matrix().toarray()
            == [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).all()
        assert (
            valid_dataset.inter_matrix().toarray()
            == [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).all()
        assert (
            test_dataset.inter_matrix().toarray()
            == [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).all()

    def test_seq_split_by_ratio(self):
        config_dict = {
            "model": "GRU4Rec",
            "dataset": "seq_dataset",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
            "eval_args": {"split": {"RS": [0.3, 0.3, 0.4]}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat[train_dataset.uid_field].numpy()
            == [1, 1, 1, 4, 2, 2, 3]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_id_list_field][:, :3].numpy()
            == [
                [1, 0, 0],
                [1, 2, 0],
                [1, 2, 3],
                [3, 0, 0],
                [4, 0, 0],
                [4, 5, 0],
                [4, 0, 0],
            ]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.iid_field].numpy()
            == [2, 3, 4, 4, 5, 6, 5]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_list_length_field].numpy()
            == [1, 2, 3, 1, 1, 2, 1]
        ).all()

        assert (
            valid_dataset.inter_feat[valid_dataset.uid_field].numpy() == [1, 1, 2]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_id_list_field][:, :5].numpy()
            == [[1, 2, 3, 4, 0], [1, 2, 3, 4, 5], [4, 5, 6, 0, 0]]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.iid_field].numpy() == [5, 6, 7]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_list_length_field].numpy()
            == [4, 5, 3]
        ).all()

        assert (
            test_dataset.inter_feat[test_dataset.uid_field].numpy() == [1, 1, 2, 3]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_id_list_field][:, :7].numpy()
            == [
                [1, 2, 3, 4, 5, 6, 0],
                [1, 2, 3, 4, 5, 6, 7],
                [4, 5, 6, 7, 0, 0, 0],
                [4, 5, 0, 0, 0, 0, 0],
            ]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.iid_field].numpy() == [7, 8, 8, 6]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_list_length_field].numpy()
            == [6, 7, 4, 2]
        ).all()

    def test_seq_benchmark(self):
        config_dict = {
            "model": "GRU4Rec",
            "dataset": "seq_benchmark",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
            "benchmark_filename": ["train", "valid", "test"],
            "alias_of_item_id": ["item_id_list"],
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        assert (
            train_dataset.inter_feat[train_dataset.uid_field].numpy()
            == [1, 1, 1, 2, 3, 3, 4]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_id_list_field][:, :3].numpy()
            == [
                [8, 0, 0],
                [8, 1, 0],
                [8, 1, 2],
                [2, 0, 0],
                [3, 0, 0],
                [3, 4, 0],
                [3, 0, 0],
            ]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.iid_field].numpy()
            == [1, 2, 3, 3, 4, 5, 4]
        ).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_list_length_field].numpy()
            == [1, 2, 3, 1, 1, 2, 1]
        ).all()

        assert (
            valid_dataset.inter_feat[valid_dataset.uid_field].numpy() == [1, 1, 3]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_id_list_field][:, :5].numpy()
            == [[8, 1, 2, 3, 0], [8, 1, 2, 3, 4], [3, 4, 5, 0, 0]]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.iid_field].numpy() == [4, 5, 6]
        ).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_list_length_field].numpy()
            == [4, 5, 3]
        ).all()

        assert (
            test_dataset.inter_feat[test_dataset.uid_field].numpy() == [1, 1, 3, 4]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_id_list_field][:, :7].numpy()
            == [
                [8, 1, 2, 3, 4, 5, 0],
                [8, 1, 2, 3, 4, 5, 6],
                [3, 4, 5, 6, 0, 0, 0],
                [3, 4, 0, 0, 0, 0, 0],
            ]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.iid_field].numpy() == [6, 7, 7, 5]
        ).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_list_length_field].numpy()
            == [6, 7, 4, 2]
        ).all()


class TestKGDataset:
    def test_kg_remap_id(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_remap_id",
            "data_path": current_path,
            "load_col": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        item_list = dataset.token2id("item_id", ["ib", "ic", "id"])
        entity_list = dataset.token2id("entity_id", ["eb", "ec", "ed", "ee", "ea"])
        assert (item_list == [1, 2, 3]).all()
        assert (entity_list == [1, 2, 3, 4, 5]).all()
        assert (dataset.inter_feat["user_id"] == [1, 2, 3]).all()
        assert (dataset.inter_feat["item_id"] == [1, 2, 3]).all()
        assert (dataset.kg_feat["head_id"] == [1, 2, 3, 4]).all()
        assert (dataset.kg_feat["tail_id"] == [5, 1, 2, 3]).all()

    def test_kg_reverse_r(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_reverse_r",
            "kg_reverse_r": True,
            "data_path": current_path,
            "load_col": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        relation_list = dataset.token2id("relation_id", ["ra", "rb", "ra_r", "rb_r"])
        assert (relation_list == [1, 2, 5, 6]).all()
        assert dataset.relation_num == 10

    def test_kg_filter_by_triple_num_in_min_entity_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "entity_kg_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 6
        assert dataset.relation_num == 6

    def test_kg_filter_by_triple_num_in_min_relation_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "relation_kg_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 7
        assert dataset.relation_num == 5

    def test_kg_filter_by_triple_num_in_max_entity_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "entity_kg_num_interval": "(-inf,3]",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 3
        assert dataset.relation_num == 3

    def test_kg_filter_by_triple_num_in_max_relation_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "relation_kg_num_interval": "(-inf,2]",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 6
        assert dataset.relation_num == 5

    def test_kg_filter_by_triple_num_in_min_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "entity_kg_num_interval": "[1,inf)",
            "relation_kg_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 7
        assert dataset.relation_num == 5

    def test_kg_filter_by_triple_num_in_complex_way(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "entity_kg_num_interval": "[1,4]",
            "relation_kg_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 7
        assert dataset.relation_num == 5


if __name__ == "__main__":
    pytest.main()
