r"""
ADMMSLIM
################################################
Reference:
    Steck et al. ADMM SLIM: Sparse Recommendations for Many Users. https://doi.org/10.1145/3336191.3371774
"""


from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


def soft_threshold(x, threshold):
    return np.maximum(x - threshold, 0.) - np.maximum(-x - threshold, 0.)


def zero_mean_columns(a):
    return a - np.mean(a, axis=0)

# https://github.com/RUCAIBox/RecBole/issues/622


def add_noise(t, mag=1e-5):
    return t + mag * torch.rand(t.shape)


class ADMMSLIM(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        X = dataset.inter_matrix(
            form='csr').astype(np.float32)

        num_users, num_items = X.shape

        lambda1 = config['lambda1']
        lambda2 = config['lambda2']
        alpha = config['alpha']
        rho = config['rho']
        k = config['k']
        positive_only = config['positive_only']
        self.center_columns = config['center_columns']
        self.item_means = X.mean(axis=0).getA1()

        if self.center_columns:
            zero_mean_X = X.toarray() - self.item_means
            G = (zero_mean_X.T @ zero_mean_X)
            # large memory cost because we need to make X dense to subtract mean, delete asap
            del zero_mean_X
        else:
            G = (X.T @ X).toarray()

        # alpha = 0 corresponds to this case (non-item specific L2 reg):
        # diag = (lambda2 + rho) * np.identity(num_items)

        diag = lambda2 * np.diag(np.power(self.item_means, alpha)) + \
            rho * np.identity(num_items)

        P = np.linalg.inv(G + diag)
        B_aux = P @ G

        # initialize
        Gamma = np.zeros_like(G)
        C = np.zeros_like(G)

        # fixed number of iterations
        for _ in range(k):
            B_tilde = B_aux + P @ (rho * C - Gamma)
            gamma = np.diag(B_tilde) / (np.diag(P) + 1e-7)
            B = B_tilde - P * gamma
            C = soft_threshold(B + Gamma/rho, lambda1/rho)
            if positive_only:
                C = np.maximum(C, 0.)
            Gamma += rho * (B - C)

        # torch doesn't support sparse tensor slicing, so will do everything with np/scipy
        self.item_similarity = C
        self.interaction_matrix = X

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        user_interactions = self.interaction_matrix[user, :].toarray()

        if self.center_columns:
            r = (((user_interactions - self.item_means) *
                  self.item_similarity[:, item].T).sum(axis=1)).flatten() + self.item_means[item]
        else:
            r = (user_interactions *
                 self.item_similarity[:, item].T).sum(axis=1).flatten()

        return add_noise(torch.from_numpy(r))

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        user_interactions = self.interaction_matrix[user, :].toarray()

        if self.center_columns:
            r = ((user_interactions - self.item_means) @
                 self.item_similarity + self.item_means).flatten()
        else:
            r = (user_interactions @ self.item_similarity).flatten()

        return add_noise(torch.from_numpy(r))
