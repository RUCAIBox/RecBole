r"""
EASE
################################################
Reference:
    Steck. "Embarrassingly Shallow Autoencoders for Sparse Data" in WWW 2019.
"""


from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


class EASE(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        X = dataset.inter_matrix(
            form='coo').astype(np.float32)

        reg_weight = config['reg_weight']  

        ## just directly calculate the entire score matrix in init
        # (can't be done incrementally)

        # gram matrix
        G = X.T @ X

        # add reg to diagonal
        G += reg_weight * sp.identity(G.shape[0])

        # convert to dense because inverse will be dense
        G = G.todense()

        # invert. this takes most of the time
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        # zero out diag
        np.fill_diagonal(B, 0.)

        S = X @ B

        self.score_matrix = torch.from_numpy(S).to(self.device)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        return self.score_matrix[user, item]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        scores = self.score_matrix[user]

        return scores.view(-1)
