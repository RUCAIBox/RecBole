r"""
SLIMElastic
################################################
Reference:
    10.1109/ICDM.2011.134
    https://www.slideshare.net/MarkLevy/efficient-slides

Reference code:
    https://github.com/KarypisLab/SLIM
    https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/master/SLIM_ElasticNet/SLIMElasticNetRecommender.py
"""


from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch
import warnings
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


# https://github.com/RUCAIBox/RecBole/issues/622
def add_noise(t, mag=1e-5):
    return t + mag * torch.rand(t.shape)


class SLIMElastic(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        X = dataset.inter_matrix(
            form='csr').astype(np.float32)

        X = X.tolil()
        self.interaction_matrix = X

        hide_item = config['hide_item']
        alpha = config['alpha']
        l1_ratio = config['l1_ratio']
        positive_only = config['positive_only']

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           positive=positive_only,
                           fit_intercept=False,
                           copy_X=False,
                           precompute=True,
                           selection='random',
                           max_iter=100,
                           tol=1e-4)

        item_coeffs = []

        # ignore ConvergenceWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            for j in range(X.shape[1]):
                # target column
                r = X[:, j]

                if hide_item:
                    # set item column to 0
                    X[:, j] = 0

                # fit the model
                model.fit(X, r.todense().getA1())

                # store the coefficients
                coeffs = model.sparse_coef_

                item_coeffs.append(coeffs)

                if hide_item:
                    # reattach column if removed
                    X[:, j] = r

        self.item_similarity = sp.vstack(item_coeffs).T

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        r = torch.from_numpy((self.interaction_matrix[user, :].multiply(
            self.item_similarity[:, item].T)).sum(axis=1).getA1())

        return add_noise(r)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        r = self.interaction_matrix[user, :] @ self.item_similarity
        r = torch.from_numpy(r.todense().getA1())

        return add_noise(r)
