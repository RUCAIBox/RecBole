# @Time   : 2020/10/19
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

import unittest

from recbole.config import Config


parameters_dict = {
    'model': 'SASRec',
    'learning_rate': 0.2,
    'topk': [50, 100],
    'epochs': 100,
}

config_file_list = ['test_config_example.yaml']


class TestConfigClass(unittest.TestCase):

    def test_default_settings(self):
        config = Config(model='BPR', dataset='ml-100k')

        self.assertEqual(config['model'], 'BPR')
        self.assertEqual(config['dataset'], 'ml-100k')

        self.assertIsInstance(config['gpu_id'], int)
        self.assertIsInstance(config['use_gpu'], bool)
        self.assertIsInstance(config['seed'], int)
        self.assertIsInstance(config['state'], str)
        self.assertIsInstance(config['data_path'], str)

        self.assertIsInstance(config['epochs'], int)
        self.assertIsInstance(config['train_batch_size'], int)
        self.assertIsInstance(config['learner'], str)
        self.assertIsInstance(config['learning_rate'], float)
        self.assertIsInstance(config['training_neg_sample_num'], int)
        self.assertIsInstance(config['eval_step'], int)
        self.assertIsInstance(config['stopping_step'], int)
        self.assertIsInstance(config['checkpoint_dir'], str)

        self.assertIsInstance(config['eval_setting'], str)
        self.assertIsInstance(config['group_by_user'], bool)
        self.assertIsInstance(config['split_ratio'], list)
        self.assertIsInstance(config['leave_one_num'], int)
        self.assertIsInstance(config['real_time_process'], bool)
        self.assertIsInstance(config['metrics'], list)
        self.assertIsInstance(config['topk'], list)
        self.assertIsInstance(config['valid_metric'], str)
        self.assertIsInstance(config['eval_batch_size'], int)

    def test_default_context_settings(self):
        config = Config(model='FM', dataset='ml-100k')

        self.assertEqual(config['eval_setting'], 'RO_RS')
        self.assertEqual(config['group_by_user'], False)
        self.assertEqual(config['metrics'], ['AUC', 'LogLoss'])
        self.assertEqual(config['valid_metric'], 'AUC')
        self.assertEqual(config['training_neg_sample_num'], 0)

    def test_default_sequential_settings(self):
        config = Config(model='SASRec', dataset='ml-100k')

        self.assertEqual(config['eval_setting'], 'TO_LS,full')

    def test_config_file_list(self):
        config = Config(model='BPR', dataset='ml-100k', config_file_list=config_file_list)

        self.assertEqual(config['model'], 'BPR')
        self.assertEqual(config['learning_rate'], 0.1)
        self.assertEqual(config['topk'], [5, 20])
        self.assertEqual(config['eval_setting'], 'TO_LS,full')

    def test_config_dict(self):
        config = Config(model='BPR', dataset='ml-100k', config_dict=parameters_dict)

        self.assertEqual(config['model'], 'BPR')
        self.assertEqual(config['learning_rate'], 0.2)
        self.assertEqual(config['topk'], [50, 100])
        self.assertEqual(config['eval_setting'], 'RO_RS,full')

    # todo: add command line test examples
    def test_priority(self):
        config = Config(model='BPR', dataset='ml-100k',
                        config_file_list=config_file_list, config_dict=parameters_dict)

        self.assertEqual(config['learning_rate'], 0.2)  # default, file, dict
        self.assertEqual(config['topk'], [50, 100])     # default, file, dict
        self.assertEqual(config['eval_setting'], 'TO_LS,full')  # default, file
        self.assertEqual(config['epochs'], 100)                 # default, dict


if __name__ == '__main__':
    unittest.main()
