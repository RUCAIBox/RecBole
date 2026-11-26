import unittest
from types import SimpleNamespace

import numpy as np
import torch

from recbole.sampler import Sampler


class TestPopularitySampler(unittest.TestCase):
    def make_sampler(self, counts, alpha=1.0):
        item_ids = np.repeat([2, 9, 42, 101], counts)
        dataset = SimpleNamespace(
            uid_field="user_id",
            iid_field="item_id",
            user_num=3,
            item_num=103,
            inter_feat={
                "user_id": torch.from_numpy(np.where(item_ids < 10, 1, 2)),
                "item_id": torch.from_numpy(item_ids),
            },
        )
        return Sampler("train", dataset, distribution="popularity", alpha=alpha)

    def test_alias_table_represents_popularity_distribution(self):
        for counts in ([1, 2, 5, 10], [3, 3, 3, 3]):
            for alpha in (0.0, 0.5, 1.0):
                with self.subTest(counts=counts, alpha=alpha):
                    sampler = self.make_sampler(counts, alpha)
                    actual = dict.fromkeys(sampler.prob, 0.0)
                    for item, probability in sampler.prob.items():
                        actual[item] += min(probability, 1.0) / len(actual)
                        if probability < 1.0:
                            if sampler.alias[item] == -1:
                                self.assertAlmostEqual(probability, 1.0)
                            else:
                                actual[sampler.alias[item]] += (
                                    1.0 - probability
                                ) / len(actual)
                    expected = np.power(counts, alpha)
                    expected /= expected.sum()
                    np.testing.assert_allclose(list(actual.values()), expected)

    def test_popularity_negatives_exclude_observed_items(self):
        sampler = self.make_sampler([1, 2, 5, 10]).set_phase("train")
        users = np.array([1, 2])
        samples = sampler.sample_by_user_ids(users, None, 50).numpy().reshape(50, 2)
        self.assertTrue(np.isin(samples[:, 0], [42, 101]).all())
        self.assertTrue(np.isin(samples[:, 1], [2, 9]).all())


if __name__ == "__main__":
    unittest.main()
