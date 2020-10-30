import os
import sys
import unittest

sys.path.append(os.getcwd())
import logging
import warnings

import numpy as np
from recbole.quick_start import run_recbole


def run_parms(parm_dict, extra_dict=None):
    config_dict = {
        'epochs': 1,
        'state': 'CRITICAL'
    }
    for name, parms in parm_dict.items():
        for parm in parms:
            config_dict[name] = parm
            if extra_dict is not None:
                config_dict.update(extra_dict)
            try:
                run_recbole(model='BPR', dataset='ml-100k', config_dict=config_dict)
            except Exception as e:
                logging.critical(f'\ntest `{name}`={parm} ... fail.\n')
                return False
    return True


class TestOverallConfig(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', ResourceWarning)

    def test_gpu_id(self):
        self.assertTrue(run_parms({'gpu_id':['0', '-1', '1']}))

    def test_use_gpu(self):
        self.assertTrue(run_parms({'use_gpu': [True, False]}))

    def test_reproducibility(self):
        self.assertTrue(run_parms({'reproducibility': [True, False]}))
        
    def test_seed(self):
        self.assertTrue(run_parms({'seed': [2021, 1024]}))

    def test_data_path(self):
        self.assertTrue(run_parms({'data_path': ['dataset/', './dataset']}))

    def test_epochs(self):
        self.assertTrue(run_parms({'epochs': [0, 1, 2]}))

    def test_train_batch_size(self):
        self.assertTrue(run_parms({'train_batch_size': [1, 2048, 200000]}))

    def test_learner(self):
        self.assertTrue(run_parms({'learner': ["adam", "sgd", "foo"]}))

    def test_learning_rate(self):
        self.assertTrue(run_parms({'learning_rate': [0, 0.001, 1e-5]}))

    def test_training_neg_sample_num(self):
        self.assertTrue(run_parms({'training_neg_sample_num': [0, 1, 2]}))

    def test_eval_step(self):
        settings = {
            'epochs':5
        }
        self.assertTrue(run_parms({'eval_step': [1, 2]}))

    def test_stopping_step(self):
        settings = {
            'epochs':100
        }
        self.assertTrue(run_parms({'stopping_step': [0, 1, 2]}))

    def test_checkpoint_dir(self):
        self.assertTrue(run_parms({'checkpoint_dir': ['saved_1/', './saved_2']}))

    def test_eval_batch_size(self):
        self.assertTrue(run_parms({'eval_batch_size': [1, 100]}))

    def test_real_time_process(self):
        self.assertTrue(run_parms({'real_time_process':[False, True]}))

    def test_topk(self):
        settings = {
            'metrics': ["Recall", "MRR", "NDCG", "Hit", "Precision"],
            'valid_metric': 'Recall@1'
        }
        self.assertTrue(run_parms({'topk':[1, [1, 3]]}, extra_dict=settings))

    def test_loss(self):
        settings = {
            'metrics':["MAE", "RMSE", "LOGLOSS", "AUC"],
            'valid_metric': 'auc',
            'eval_setting': 'RO_RS, uni100'
        }
        self.assertTrue(run_parms({'topk':{None, 1}}, extra_dict=settings))

    def test_metric(self):
        settings = {
            'topk':3,
            'valid_metric': 'Recall@3'
        }
        self.assertTrue(run_parms({'metrics':["Recall", ["Recall", "MRR", "NDCG", "Hit", "Precision"]]}, extra_dict=settings))

    def test_split_ratio(self):
        settings = {
            'leave_one_num':None
        }

        self.assertTrue(run_parms({'split_ratio':[[0.8, 0.2], [0.8, 0.1, 0.1], [7, 1, 1, 1], [16, 2, 2]]}))

    def test_leave_one_num(self):
        settings = {
            'split_ratio':None
        }

        self.assertTrue(run_parms({'leave_one_num':[1, 2, 3]}))
        
    def test_group_by_user(self):
        self.assertTrue(run_parms({'group_by_user':[True, False]}))



if __name__ == '__main__':
    # suite = unittest.TestSuite()
    # suite.addTest(TestOverallConfig('test_split_ratio'))
    # runner = unittest.TextTestRunner(verbosity=2)
    # runner.run(suite)
    unittest.main(verbosity=1)
