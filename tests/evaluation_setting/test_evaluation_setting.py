# -*- coding: utf-8 -*-
# @Time   : 2020/10/27
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/1
# @Author : Xingyu Pan
# @Email  : xy_pan@foxmail.com
import os
import unittest

from recbole.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, "../model/test_model.yaml")]


class TestGeneralRecommender(unittest.TestCase):
    def test_rols_full(self):
        config_dict = {
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "mode": "full",
            },
            "model": "BPR",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )

    def test_tols_full(self):
        config_dict = {
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "mode": "full",
            },
            "model": "BPR",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )

    def test_tors_full(self):
        config_dict = {
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "RO",
                "mode": "full",
            },
            "model": "BPR",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )

    def test_rors_uni100(self):
        config_dict = {
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "RO",
                "mode": "uni100",
            },
            "model": "BPR",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )

    def test_tols_uni100(self):
        config_dict = {
            "eval_setting": "TO_LS,uni100",
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "mode": "full",
            },
            "model": "BPR",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )

    def test_rols_uni100(self):
        config_dict = {
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "mode": "uni100",
            },
            "model": "BPR",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )

    def test_tors_uni100(self):
        config_dict = {
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "uni100",
            },
            "model": "BPR",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )


class TestContextRecommender(unittest.TestCase):
    def test_tors(self):
        config_dict = {
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "labeled",
            },
            "threshold": {"rating": 4},
            "model": "FM",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )


class TestSequentialRecommender(unittest.TestCase):
    def test_tols_uni100(self):
        config_dict = {
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "mode": "uni100",
            },
            "model": "FPMC",
        }
        objective_function(
            config_dict=config_dict, config_file_list=config_file_list, saved=False
        )


if __name__ == "__main__":
    unittest.main()
