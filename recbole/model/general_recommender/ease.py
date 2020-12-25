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


def scipy_to_sparse_tensor(A):
    # https://stackoverflow.com/a/50665264/7367514
    C = A.tocoo()

    values = C.data
    indices = np.vstack((C.row, C.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = C.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


class EASE(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        X = dataset.inter_matrix(
            form='csr').astype(np.float32)

        reg_weight = config['reg_weight']

        # just directly calculate the entire score matrix in init
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

        # instead of computing and storing the entire score matrix, just store B and compute the scores on demand
        # more memory efficient for a larger number of users
        # but if there's a large number of items not much one can do:
        # still have to compute B all at once
        # S = X @ B
        # self.score_matrix = torch.from_numpy(S).to(self.device)

        # torch doesn't support sparse tensor slicing, so will do everything with np/scipy
        self.item_similarity = B
        self.interaction_matrix = X

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        return torch.from_numpy((self.interaction_matrix[user, :].multiply(self.item_similarity[:, item].T)).sum(axis=1).getA1())

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        r = self.interaction_matrix[user, :] @ self.item_similarity
        return torch.from_numpy(r.flatten())
