# -*- coding: utf-8 -*-
# @Time   : 2022/7/30
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com


import logging
import os
import math
import numpy as np

import pytest

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from recbole.data.transform import construct_transform

current_path = os.path.dirname(os.path.realpath(__file__))


def new_dataset(config_dict=None, config_file_list=None):
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logging.basicConfig(level=logging.ERROR)
    return create_dataset(config)


def split_dataset(config_dict=None, config_file_list=None):
    dataset = new_dataset(config_dict=config_dict, config_file_list=config_file_list)
    return dataset.build()


def new_transform(config_dict=None, config_file_list=None):
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logging.basicConfig(level=logging.ERROR)
    return construct_transform(config)


class TestTransform:
    def test_mask_itemseq(self):
        config_dict = {
            "model": "BERT4Rec",
            "dataset": "seq_dataset",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
            "transform": "mask_itemseq",
            "mask_ratio": 1.0,
            "ft_ratio": 0.0,
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        transform = new_transform(config_dict=config_dict)

        train_transform_interaction = transform(train_dataset, train_dataset.inter_feat)
        assert (
            train_transform_interaction[transform.MASK_ITEM_SEQ][:, :5].numpy()
            == [
                [9, 0, 0, 0, 0],
                [9, 9, 0, 0, 0],
                [9, 9, 9, 0, 0],
                [9, 9, 9, 9, 0],
                [9, 9, 9, 9, 9],
                [9, 0, 0, 0, 0],
                [9, 0, 0, 0, 0],
                [9, 9, 0, 0, 0],
                [9, 0, 0, 0, 0],
            ]
        ).all()

        valid_transform_interaction = transform(valid_dataset, valid_dataset.inter_feat)
        assert (
            valid_transform_interaction[transform.MASK_ITEM_SEQ][:, :6].numpy()
            == [[9, 9, 9, 9, 9, 9], [9, 9, 9, 0, 0, 0]]
        ).all()

        test_transform_interaction = transform(test_dataset, test_dataset.inter_feat)
        assert (
            test_transform_interaction[transform.MASK_ITEM_SEQ][:, :7].numpy()
            == [[9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 0, 0, 0], [9, 9, 0, 0, 0, 0, 0]]
        ).all()

    def test_inverse_itemseq(self):
        config_dict = {
            "model": "SHAN",
            "dataset": "seq_dataset",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
            "transform": "inverse_itemseq",
            "eval_args": {"split": {"RS": [0.3, 0.3, 0.4]}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        transform = new_transform(config_dict=config_dict)

        train_transform_interaction = transform(train_dataset, train_dataset.inter_feat)
        assert (
            train_transform_interaction[transform.INVERSE_ITEM_SEQ][:, -3:].numpy()
            == [
                [0, 0, 1],
                [0, 1, 2],
                [1, 2, 3],
                [0, 0, 3],
                [0, 0, 4],
                [0, 4, 5],
                [0, 0, 4],
            ]
        ).all()

        valid_transform_interaction = transform(valid_dataset, valid_dataset.inter_feat)
        assert (
            valid_transform_interaction[transform.INVERSE_ITEM_SEQ][:, -5:].numpy()
            == [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [0, 0, 4, 5, 6]]
        ).all()

        test_transform_interaction = transform(test_dataset, test_dataset.inter_feat)
        assert (
            test_transform_interaction[transform.INVERSE_ITEM_SEQ][:, -7:].numpy()
            == [
                [0, 1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 6, 7],
                [0, 0, 0, 4, 5, 6, 7],
                [0, 0, 0, 0, 0, 4, 5],
            ]
        ).all()

    def test_crop_itemseq(self):
        eta = 0.6
        config_dict = {
            "model": "GRU4Rec",
            "dataset": "seq_dataset",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
            "transform": "crop_itemseq",
            "eta": eta,
            "eval_args": {"split": {"RS": [0.3, 0.3, 0.4]}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        transform = new_transform(config_dict=config_dict)

        train_transform_interaction = transform(train_dataset, train_dataset.inter_feat)
        for crop_seq, length in zip(
            train_transform_interaction[transform.CROP_ITEM_SEQ],
            train_transform_interaction[transform.ITEM_SEQ_LEN],
        ):
            crop_len = math.floor(length * eta)
            last_seq = np.zeros(length - crop_len)

            assert (crop_seq[crop_len:length].numpy() == last_seq).all()

        valid_transform_interaction = transform(valid_dataset, valid_dataset.inter_feat)
        for crop_seq, length in zip(
            valid_transform_interaction[transform.CROP_ITEM_SEQ],
            valid_transform_interaction[transform.ITEM_SEQ_LEN],
        ):
            crop_len = math.floor(length * eta)
            last_seq = np.zeros(length - crop_len)

            assert (crop_seq[crop_len:length].numpy() == last_seq).all()

        test_transform_interaction = transform(test_dataset, test_dataset.inter_feat)
        for crop_seq, length in zip(
            test_transform_interaction[transform.CROP_ITEM_SEQ],
            test_transform_interaction[transform.ITEM_SEQ_LEN],
        ):
            crop_len = math.floor(length * eta)
            last_seq = np.zeros(length - crop_len)

            assert (crop_seq[crop_len:length].numpy() == last_seq).all()

    def test_reorder_itemseq(self):
        beta = 0.9
        config_dict = {
            "model": "GRU4Rec",
            "dataset": "reorder_dataset",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
            "transform": "reorder_itemseq",
            "beta": beta,
            "eval_args": {"split": {"RS": [0.3, 0.3, 0.4]}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(
            config_dict=config_dict
        )
        transform = new_transform(config_dict=config_dict)

        train_transform_interaction = transform(train_dataset, train_dataset.inter_feat)
        reorder_item_seq = train_transform_interaction[transform.REORDER_ITEM_SEQ]
        item_seq = train_dataset.inter_feat[train_dataset.item_id_list_field]
        assert (reorder_item_seq.numpy() != item_seq.numpy()).any()

        valid_transform_interaction = transform(valid_dataset, valid_dataset.inter_feat)
        reorder_item_seq = valid_transform_interaction[transform.REORDER_ITEM_SEQ]
        item_seq = valid_dataset.inter_feat[valid_dataset.item_id_list_field]
        assert (reorder_item_seq.numpy() != item_seq.numpy()).any()

        test_transform_interaction = transform(test_dataset, test_dataset.inter_feat)
        reorder_item_seq = test_transform_interaction[transform.REORDER_ITEM_SEQ]
        item_seq = test_dataset.inter_feat[test_dataset.item_id_list_field]
        assert (reorder_item_seq.numpy() != item_seq.numpy()).any()


if __name__ == "__main__":
    pytest.main()
