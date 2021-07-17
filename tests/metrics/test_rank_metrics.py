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
from recbole.config import Config
from recbole.evaluator import metrics_dict, Collector

parameters_dict = {
    'model': 'BPR',
    'eval_args': {'split':{'RS':[0.8,0.1,0.1]}, 'order': 'RO', 'mode': 'uni100'},
    'metric_decimal_place': 4,
}

config = Config('BPR', 'ml-1m', config_dict=parameters_dict)


class MetricsTestCases(object):
    user_len_list0 = np.array([2, 3, 5])
    pos_len_list0 = np.array([1, 2, 3])
    pos_rank_sum0 = np.array([1, 4, 9])

    user_len_list1 = np.array([3, 6, 4])
    pos_len_list1 = np.array([1, 0, 4])
    pos_rank_sum1 = np.array([3, 0, 6])


def get_metric_result(name, case=0):
    Metric = metrics_dict[name](config)
    return Metric.metric_info(
                getattr(MetricsTestCases, f'pos_rank_sum{case}'),
                getattr(MetricsTestCases, f'user_len_list{case}'),
                getattr(MetricsTestCases, f'pos_len_list{case}'))


class TestRankMetrics(unittest.TestCase):
    def test_gauc(self):
        name = 'gauc'
        self.assertEqual(get_metric_result(name, case=0), (1 * ((2 - (1 - 1) / 2 - 1 / 1) / (2 - 1)) +
                                                           2 * ((3 - (2 - 1) / 2 - 4 / 2) / (3 - 2)) +
                                                           3 * ((5 - (3 - 1) / 2 - 9 / 3) / (5 - 3)))
                         / (1 + 2 + 3))
        self.assertEqual(get_metric_result(name, case=1), (3 - 0 - 3 / 1) / (3 - 1))


if __name__ == "__main__":
    unittest.main()
