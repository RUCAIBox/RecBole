# -*- coding: utf-8 -*-
# @Time   : 2024/12/01
# @Author : Markus Hoefling
# @Email  : markus.hoefling01@gmail.com

r"""
ALS
################################################
Reference 1:
    Hu, Y., Koren, Y., & Volinsky, C. (2008). "Collaborative Filtering for Implicit Feedback Datasets." In ICDM 2008.

Reference 2:
    Frederickson, Ben, "Implicit 0.7.2", code: https://github.com/benfred/implicit, readthedocs: https://benfred.github.io/implicit/
"""

import numpy as np
import threadpoolctl
import torch
from implicit.als import AlternatingLeastSquares
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
threadpoolctl.threadpool_limits(1, "blas") # Due to a warning that occurred while running the ALS algorithm

class ALS(GeneralRecommender):
    r"""
    ALS is a matrix factorization model implemented using the Alternating Least Squares (ALS) method
    from the `implicit` library (https://benfred.github.io/implicit/).
    This model optimizes the embeddings through the Alternating Least Squares algorithm.
    """

    input_type = InputType.POINTWISE
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(ALS, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.regularization = config['regularization']
        self.alpha = config['alpha']
        self.iterations = config['epochs']

        # define model
        self.model = AlternatingLeastSquares(
            factors=self.embedding_size,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=1, # iterations are done by the trainer via epochs
            use_cg=True,
            calculate_training_loss=True,
            num_threads=24,
            random_state=42
        )

        # initialize embeddings
        self.user_embeddings = np.random.rand(self.n_users, self.embedding_size)
        self.item_embeddings = np.random.rand(self.n_items, self.embedding_size)

        # fake embeddings for optimizer initialization
        self.fake_parameter = torch.nn.Parameter(torch.zeros(1))

    def get_user_embedding(self, user):
        return torch.tensor(self.user_embeddings[user])

    def get_item_embedding(self, item):
        return torch.tensor(self.item_embeddings[item])

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def _callback(self, iteration, time, loss):
        self._loss = loss

    def calculate_loss(self, interactions):
        self.model.fit(interactions, show_progress=False, callback=self._callback)
        self.user_embeddings = self.model.user_factors
        self.item_embeddings = self.model.item_factors
        return self._loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.dot(user_e, item_e)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = torch.tensor(self.model.item_factors)
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)