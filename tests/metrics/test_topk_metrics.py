# -*- encoding: utf-8 -*-
# @Time    :   2020/11/1
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :    2021/7/2, 2021/7/5
# @Author  :    Zihan Lin, Zhichao Feng
# @email   :    zhlin@ruc.edu.cn, fzcbupt@gmail.com

import os
import sys
import unittest

sys.path.append(os.getcwd())
import numpy as np
from recbole.config import Config
from recbole.evaluator.register import metrics_dict

parameters_dict = {
    "topk": [10],
    "metric_decimal_place": 4,
}

config = Config("BPR", "ml-1m", config_dict=parameters_dict)
pos_idx = np.array(
    [
        [0, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
    ]
)
pos_len = np.array([1, 3, 4, 2])

item_matrix = np.array([[5, 7, 3], [4, 5, 2], [2, 3, 5], [1, 4, 6], [5, 3, 7]])

num_items = 8

item_count = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


class TestTopKMetrics(unittest.TestCase):
    def test_hit(self):
        name = "hit"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1]]).tolist(),
        )

    def test_ndcg(self):
        name = "ndcg"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [
                        1,
                        (1 / np.log2(2) / (1 / np.log2(2) + 1 / np.log2(3))),
                        (
                            (1 / np.log2(2) + 1 / np.log2(4))
                            / (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4))
                        ),
                    ],
                    [0, 0, (1 / np.log2(4) / (1 / np.log2(2) + 1 / np.log2(3)))],
                ]
            ).tolist(),
        )

    def test_mrr(self):
        name = "mrr"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1 / 3]]).tolist(),
        )

    def test_map(self):
        name = "map"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, (1 / 2), (1 / 3) * ((1 / 1) + (2 / 3))],
                    [0, 0, (1 / 3) * (1 / 2)],
                ]
            ).tolist(),
        )

    def test_recall(self):
        name = "recall"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array(
                [[0, 0, 0], [1 / 3, 2 / 3, 3 / 3], [1 / 4, 1 / 4, 2 / 4], [0, 0, 1 / 2]]
            ).tolist(),
        )

    def test_precision(self):
        name = "precision"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array(
                [[0, 0, 0], [1 / 1, 2 / 2, 3 / 3], [1 / 1, 1 / 2, 2 / 3], [0, 0, 1 / 3]]
            ).tolist(),
        )

    def test_itemcoverage(self):
        name = "itemcoverage"
        Metric = metrics_dict[name](config)
        self.assertEqual(Metric.get_coverage(item_matrix, num_items), 7 / 8)

    def test_averagepopularity(self):
        name = "averagepopularity"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(Metric.get_pop(item_matrix, item_count)).tolist(),
            np.array(
                [
                    [4 / 1, 4 / 2, 6 / 3],
                    [3 / 1, 7 / 2, 8 / 3],
                    [1 / 1, 3 / 2, 7 / 3],
                    [0 / 1, 3 / 2, 8 / 3],
                    [4 / 1, 6 / 2, 6 / 3],
                ]
            ).tolist(),
        )

    def test_giniindex(self):
        name = "giniindex"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.get_gini(item_matrix, num_items),
            ((-7) * 0 + (-5) * 1 + (-3) * 1 + (-1) * 2 + 1 * 2 + 3 * 2 + 5 * 3 + 7 * 4)
            / (8 * (3 * 5)),
        )

    def test_shannonentropy(self):
        name = "shannonentropy"
        Metric = metrics_dict[name](config)
        self.assertAlmostEqual(
            Metric.get_entropy(item_matrix),
            -np.sum(
                [
                    1 / 15 * np.log(1 / 15),
                    2 / 15 * np.log(2 / 15),
                    3 / 15 * np.log(3 / 15),
                    2 / 15 * np.log(2 / 15),
                    4 / 15 * np.log(4 / 15),
                    1 / 15 * np.log(1 / 15),
                    2 / 15 * np.log(2 / 15),
                ]
            ),
            delta=1e-15,
        )

    def test_tailpercentage(self):
        name = "tailpercentage"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(Metric.get_tail(item_matrix, item_count)).tolist(),
            np.array(
                [
                    [0 / 1, 0 / 2, 0 / 3],
                    [0 / 1, 0 / 2, 0 / 3],
                    [0 / 1, 0 / 2, 0 / 3],
                    [1 / 1, 1 / 2, 1 / 3],
                    [0 / 1, 0 / 2, 0 / 3],
                ]
            ).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
