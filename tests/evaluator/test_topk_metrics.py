import os
import sys
import unittest

sys.path.append(os.getcwd())
import numpy as np
from recbole.evaluator.metrics import metrics_dict

pos_idx = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 1],
    [0, 0, 1],
])
pos_len = np.array([1, 3, 4, 2])


def get_result(name):
    func = metrics_dict[name]
    return func(pos_idx, pos_len)


class TestTopKMetrics(unittest.TestCase):
    def test_hit(self):
        name = 'hit'
        self.assertEqual(
            get_result(name).tolist(),
            np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1]]).tolist())

    def test_ndcg(self):
        name = 'ndcg'
        self.assertEqual(
            get_result(name).tolist(),
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
        self.assertEqual(
            get_result(name).tolist(),
            np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0,
                                                        1 / 3]]).tolist())

    def test_map(self):
        name = 'map'
        self.assertEqual(
            get_result(name).tolist(),
            np.array([[0, 0, 0], [1, 1, 1],
                      [1, (1 / 2), (1 / 3) * ((1 / 1) + (2 / 3))],
                      [0, 0, (1 / 3) * (1 / 2)]]).tolist())

    def test_recall(self):
        name = 'recall'
        self.assertEqual(
            get_result(name).tolist(),
            np.array([[0, 0, 0], [1 / 3, 2 / 3, 3 / 3], [1 / 4, 1 / 4, 2 / 4],
                      [0, 0, 1 / 2]]).tolist())

    def test_precision(self):
        name = 'precision'
        self.assertEqual(
            get_result(name).tolist(),
            np.array([[0, 0, 0], [1 / 1, 2 / 2, 3 / 3], [1 / 1, 1 / 2, 2 / 3],
                      [0, 0, 1 / 3]]).tolist())


if __name__ == "__main__":
    unittest.main()
