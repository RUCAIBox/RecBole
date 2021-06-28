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
from recbole.evaluator.metrics import metrics_dict

parameters_dict = {
    'topk': [10],
    'metric_decimal_place': 4,
}

config = Config('BPR', 'ml-1m', config_dict=parameters_dict)
pos_idx = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 1],
    [0, 0, 1],
])
pos_len = np.array([1, 3, 4, 2])


class TestTopKMetrics(unittest.TestCase):
    def test_hit(self):
        name = 'hit'
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1]]).tolist())

    def test_ndcg(self):
        name = 'ndcg'
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array([[0, 0, 0], [1, 1, 1],
                      [
                          1,
                          (1 / np.log2(2) / (1 / np.log2(2) + 1 / np.log2(3))),
                          ((1 / np.log2(2) + 1 / np.log2(4)) / (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)))
                      ],
                      [
                          0,
                          0,
                          (1 / np.log2(4) / (1 / np.log2(2) + 1 / np.log2(3)))
                      ]]).tolist())

    def test_mrr(self):
        name = 'mrr'
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0,
                                                        1 / 3]]).tolist())

    def test_map(self):
        name = 'map'
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array([[0, 0, 0], [1, 1, 1],
                      [1, (1 / 2), (1 / 3) * ((1 / 1) + (2 / 3))],
                      [0, 0, (1 / 3) * (1 / 2)]]).tolist())

    def test_recall(self):
        name = 'recall'
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array([[0, 0, 0], [1 / 3, 2 / 3, 3 / 3], [1 / 4, 1 / 4, 2 / 4],
                      [0, 0, 1 / 2]]).tolist())

    def test_precision(self):
        name = 'precision'
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array([[0, 0, 0], [1 / 1, 2 / 2, 3 / 3], [1 / 1, 1 / 2, 2 / 3],
                      [0, 0, 1 / 3]]).tolist())


if __name__ == "__main__":
    unittest.main()
