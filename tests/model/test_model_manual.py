# -*- coding: utf-8 -*-
# @Time   : 2020/11/18
# @Author : Xingyu Pan
# @Email  : panxy@ruc.edu.cn


import os
import unittest

from recbole.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]


class TestContextRecommender(unittest.TestCase):
    # todo: more complex context information should be test, such as criteo dataset
  
    def test_dcn(self):
        config_dict = {
            'model': 'DCN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)

    # def test_din(self):
    #     config_dict = {
    #         'model': 'DIN',
    #     }
    #     objective_function(config_dict=config_dict,
    #                        config_file_list=config_file_list, saved=False)


class TestSequentialRecommender(unittest.TestCase):

    def test_bert4rec(self):
        config_dict = {
            'model': 'BERT4Rec',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)

    def test_fdsa(self):
        config_dict = {
            'model': 'FDSA',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)

    # def test_gru4reckg(self):
    #     config_dict = {
    #         'model': 'GRU4RecKG',
    #     }
    #     objective_function(config_dict=config_dict,
    #                        config_file_list=config_file_list, saved=False)

    def test_s3rec(self):
        config_dict = {
            'model': 'S3Rec',
            'train_stage': 'pretrain',
            'save_step': 1,
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)

        config_dict = {
            'model': 'S3Rec',
            'train_stage': 'finetune',
            'pre_model_path': './saved/S3Rec-test-1.pth',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)


class TestSequentialRecommender2(unittest.TestCase):

    def test_bert4rec(self):
        config_dict = {
            'model': 'BERT4Rec',
            'loss_type': 'BPR',
            'hidden_act': 'swish'
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)

   
if __name__ == '__main__':
    unittest.main()
