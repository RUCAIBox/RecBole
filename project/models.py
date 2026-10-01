"""Models defined in this project (outside RecBole).

To use one, set `model: <ClassName>` in configs/models/<experiment>.yaml and add the class to
PROJECT_MODELS below.
"""

import numpy as np
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class TrainPop(GeneralRecommender):
    r"""Most-popular baseline: score(u, i) = (# training interactions of i) / (max over items).

    Same scoring as RecBole's Pop, but the counts are read directly from the training
    interaction matrix. RecBole's Pop accumulates counts batch by batch with
    `cnt[item] = cnt[item] + 1`, which counts an item at most once per batch and also counts
    the sampled negative items, so its "popularity" is not the true interaction count.
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(TrainPop, self).__init__(config, dataset)
        counts = np.asarray(dataset.inter_matrix(form="csr").sum(axis=0), dtype=np.float32).ravel()
        self.register_buffer("item_cnt", torch.from_numpy(counts).to(self.device))
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1)).to(self.device)

    def predict(self, interaction):
        return self.item_cnt[interaction[self.ITEM_ID]] / self.item_cnt.max()

    def full_sort_predict(self, interaction):
        n_users = interaction[self.USER_ID].shape[0]
        return (self.item_cnt / self.item_cnt.max()).repeat(n_users)


PROJECT_MODELS = {cls.__name__: cls for cls in [TrainPop]}
