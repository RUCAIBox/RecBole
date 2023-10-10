# -*- coding: utf-8 -*-
# @Time   : 2020/11/18
# @Author : Xingyu Pan
# @Email  : panxy@ruc.edu.cn


import os
import unittest

from recbole.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, "test_model.yaml")]


def quick_test(config_dict):
    objective_function(
        config_dict=config_dict, config_file_list=config_file_list, saved=False
    )


class TestSequentialRecommender(unittest.TestCase):
    # def test_gru4reckg(self):
    #     config_dict = {
    #         'model': 'GRU4RecKG',
    #     }
    #     quick_test(config_dict)

    def test_s3rec(self):
        config_dict = {
            "model": "S3Rec",
            "train_stage": "pretrain",
            "save_step": 1,
            "train_neg_sample_args": None,
        }
        quick_test(config_dict)

        config_dict = {
            "model": "S3Rec",
            "train_stage": "finetune",
            "pre_model_path": "./saved/S3Rec-test-1.pth",
            "train_neg_sample_args": None,
        }
        quick_test(config_dict)


if __name__ == "__main__":
    unittest.main()
