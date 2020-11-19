# -*- coding: utf-8 -*-
# @Time   : 2020/10/27
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/11/17
# @Author : Xingyu Pan
# @Email  : panxy@ruc.edu.cn
import os
import unittest

from recbole.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, '../model/test_model.yaml')]


class TestGeneralRecommender(unittest.TestCase):

    def test_rols_full(self):
        config_dict = {
            'eval_setting': 'RO_LS,full',
            'model': 'BPR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        '''
        config_dict = {
            'eval_setting': 'RO_LS,full',
            'model': 'NeuMF',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        config_dict = {
            'eval_setting': 'RO_LS,full',
            'model': 'FISM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        config_dict = {
            'eval_setting': 'RO_LS,full',
            'model': 'LightGCN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        '''
    def test_tols_full(self):
        config_dict = {
            'eval_setting': 'TO_LS,full',
            'model': 'BPR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        '''
        config_dict = {
            'eval_setting': 'TO_LS,full',
            'model': 'NeuMF',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        config_dict = {
            'eval_setting': 'TO_LS,full',
            'model': 'FISM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        config_dict = {
            'eval_setting': 'TO_LS,full',
            'model': 'LightGCN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        '''
    def test_tors_full(self):
        config_dict = {
            'eval_setting': 'TO_RS,full',
            'model': 'BPR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS,full',
        #     'model': 'NeuMF',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS,full',
        #     'model': 'FISM',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS,full',
        #     'model': 'LightGCN',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)

    def test_rors_uni100(self):
        config_dict = {
            'eval_setting': 'RO_RS,uni100',
            'model': 'BPR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'RO_RS,uni100',
        #     'model': 'NeuMF',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'RO_RS,uni100',
        #     'model': 'FISM',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'RO_RS,uni100',
        #     'model': 'LightGCN',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)

    def test_tols_uni100(self):
        config_dict = {
            'eval_setting': 'TO_LS,uni100',
            'model': 'BPR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_LS,uni100',
        #     'model': 'NeuMF',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_LS,uni100',
        #     'model': 'FISM',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_LS,uni100',
        #     'model': 'LightGCN',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)

    def test_rols_uni100(self):
        config_dict = {
            'eval_setting': 'RO_LS,uni100',
            'model': 'BPR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'RO_LS,uni100',
        #     'model': 'NeuMF',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'RO_LS,uni100',
        #     'model': 'FISM',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'RO_LS,uni100',
        #     'model': 'LightGCN',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)

    def test_tors_uni100(self):
        config_dict = {
            'eval_setting': 'TO_RS,uni100',
            'model': 'BPR',
        }
        objective_function(config_dict=config_dict,
                            config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS,uni100',
        #     'model': 'NeuMF',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS,uni100',
        #     'model': 'FISM',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS,uni100',
        #     'model': 'LightGCN',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)


class TestContextRecommender(unittest.TestCase):

    def test_tors(self):
        config_dict = {
            'eval_setting': 'TO_RS',
            'model': 'FM',
        }
        objective_function(config_dict=config_dict,
                            config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS',
        #     'model': 'DeepFM',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS',
        #     'model': 'DSSM',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_RS',
        #     'model': 'AutoInt',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)


class TestSequentialRecommender(unittest.TestCase):

    def test_tols_uni100(self):
        config_dict = {
            'eval_setting': 'TO_LS,uni100',
            'model': 'FPMC',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_LS,uni100',
        #     'model': 'SASRec',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_LS,uni100',
        #     'model': 'GRU4RecF',
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)
        # config_dict = {
        #     'eval_setting': 'TO_LS,uni100',
        #     'model': 'Caser',
        #     'MAX_ITEM_LIST_LENGTH': 10,
        #     'reproducibility': False,
        # }
        # objective_function(config_dict=config_dict,
        #                    config_file_list=config_file_list, saved=False)


if __name__ == '__main__':
    unittest.main()
