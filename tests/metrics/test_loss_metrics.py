# -*- encoding: utf-8 -*-
# @Time    :   2020/11/1
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com


import os
import sys
import unittest

sys.path.append(os.getcwd())
import numpy as np
from recbole.config import Config
from recbole.evaluator.register import metrics_dict

parameters_dict = {
    'metric_decimal_place': 4,
}

config = Config('BPR', 'ml-1m', config_dict=parameters_dict)


class TestCases(object):
    preds_0 = np.array([0.1, 0.9, 0.2, 0.3])
    trues_0 = np.array([1, 0, 1, 1])

    preds_1 = np.array([0.7, 0.5, 0.6, 0.2])
    trues_1 = np.array([0, 1, 1, 0])


def get_result(name, case=0):
    Metric = metrics_dict[name](config)
    return Metric.metric_info(
        getattr(TestCases, f'preds_{case}'),
        getattr(TestCases, f'trues_{case}'))


class TestLossMetrics(unittest.TestCase):
    def test_auc(self):
        name = 'auc'
        self.assertEqual(get_result(name, case=0), 0)
        self.assertEqual(get_result(name, case=1), 2 / (2 * 2))

    def test_rmse(self):
        name = 'rmse'
        self.assertEqual(get_result(name, case=0),
                         np.sqrt((0.9 ** 2 + 0.9 ** 2 + 0.8 ** 2 + 0.7 ** 2) / 4))
        self.assertEqual(get_result(name, case=1),
                         np.sqrt((0.7 ** 2 + 0.5 ** 2 + 0.4 ** 2 + 0.2 ** 2) / 4))

    def test_logloss(self):
        name = 'logloss'
        self.assertAlmostEqual(
            get_result(name, case=0),
            (-np.log(0.1) - np.log(0.2) - np.log(0.3) - np.log(0.1)) / 4)
        self.assertAlmostEqual(
            get_result(name, case=1),
            (-np.log(0.5) - np.log(0.6) - np.log(0.3) - np.log(0.8)) / 4)

    def test_mae(self):
        name = 'mae'
        self.assertEqual(get_result(name, case=0), (0.9 + 0.9 + 0.8 + 0.7) / 4)
        self.assertEqual(get_result(name, case=1), (0.7 + 0.5 + 0.4 + 0.2) / 4)


if __name__ == "__main__":
    unittest.main()
