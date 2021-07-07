# -*- encoding: utf-8 -*-
# @Time    :   2020/12/21
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

# UPDATE:
# @Time   : 2021/7/1
# @Author : Xingyu Pan
# @Email  : xy_pan@foxmail.com

import os
import sys
import unittest

sys.path.append(os.getcwd())
import numpy as np
import torch
from recbole.config import Config
from recbole.data.interaction import Interaction
from recbole.evaluator.metrics import metrics_dict
from recbole.evaluator.evaluators import RankEvaluator

parameters_dict = {
    'model': 'BPR',
    'eval_args': {'split':{'RS':[0.8,0.1,0.1]}, 'order': 'RO', 'mode': 'uni100'}
}


class MetricsTestCases(object):
    user_len_list0 = np.array([2, 3, 5])
    pos_len_list0 = np.array([1, 2, 3])
    pos_rank_sum0 = np.array([1, 4, 9])

    user_len_list1 = np.array([3, 6, 4])
    pos_len_list1 = np.array([1, 0, 4])
    pos_rank_sum1 = np.array([3, 0, 6])


class CollectTestCases(object):
    interaction0 = Interaction({}, [0, 2, 3, 4], [2, 3, 4, 5])
    scores_tensor0 = torch.Tensor([0.1, 0.2,
                                   0.1, 0.1, 0.2,
                                   0.2, 0.2, 0.2, 0.2,
                                   0.3, 0.2, 0.1, 0.4, 0.3])


def get_metric_result(name, case=0):
    func = metrics_dict[name]
    return func(getattr(MetricsTestCases, f'user_len_list{case}'),
                getattr(MetricsTestCases, f'pos_len_list{case}'),
                getattr(MetricsTestCases, f'pos_rank_sum{case}'))


def get_collect_result(evaluator, case=0):
    func = evaluator.collect
    return func(getattr(CollectTestCases, f'interaction{case}'),
                getattr(CollectTestCases, f'scores_tensor{case}'))


class TestRankMetrics(unittest.TestCase):
    def test_gauc(self):
        name = 'gauc'
        self.assertEqual(get_metric_result(name, case=0), (1 * ((2 - (1 - 1) / 2 - 1 / 1) / (2 - 1)) +
                                                           2 * ((3 - (2 - 1) / 2 - 4 / 2) / (3 - 2)) +
                                                           3 * ((5 - (3 - 1) / 2 - 9 / 3) / (5 - 3)))
                         / (1 + 2 + 3))
        self.assertEqual(get_metric_result(name, case=1), (3 - 0 - 3 / 1) / (3 - 1))

    def test_collect(self):
        config = Config('BPR', 'ml-100k', config_dict=parameters_dict)
        metrics = ['GAUC']
        rank_evaluator = RankEvaluator(config, metrics)
        self.assertEqual(get_collect_result(rank_evaluator, case=0).squeeze().cpu().numpy().tolist(),
                         np.array([0, (2 + 3) / 2 * 2, (1 + 2 + 3 + 4) / 4 * 3, 1 + (2 + 3) / 2 + 4 + 5]).tolist())


if __name__ == "__main__":
    unittest.main()
