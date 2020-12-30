# -*- coding: utf-8 -*-
# @Time   : 2020/11/18
# @Author : Xingyu Pan
# @Email  : panxy@ruc.edu.cn


import os
import unittest

from recbole.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]


# class TestSequentialRecommender(unittest.TestCase):
#
#     def test_gru4reckg(self):
#         config_dict = {
#             'model': 'GRU4RecKG',
#         }
#         objective_function(config_dict=config_dict,
#                            config_file_list=config_file_list, saved=False)


if __name__ == '__main__':
    unittest.main()
