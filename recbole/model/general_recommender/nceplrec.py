r"""
NCE-PLRec
################################################
Reference:
"""


from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.utils.extmath import randomized_svd

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


class NCEPLRec(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        R = dataset.inter_matrix(
            form='csr').astype(np.float32)

        beta = config['beta']
        rank = int(config['rank'])
        reg_weight = config['reg_weight']
        seed = config['seed']

        # just directly calculate the entire score matrix in init
        # (can't be done incrementally)
        num_users, num_items = R.shape

        item_popularities = R.sum(axis=0)

        D_rows = []
        for i in range(num_users):
            row_index, col_index = R[i].nonzero()
            if len(row_index) > 0:
                values = item_popularities[:, col_index].getA1()
                # note this is a slight variation of what's in the paper, for convenience
                # see https://github.com/wuga214/NCE_Projected_LRec/issues/38
                values = np.maximum(
                    np.log(num_users/np.power(values, beta)), 0)
                D_rows.append(sp.coo_matrix(
                    (values, (row_index, col_index)), shape=(1, num_items)))
            else:
                D_rows.append(sp.coo_matrix((1, num_items)))

        D = sp.vstack(D_rows)

        _, sigma, Vt = randomized_svd(D, n_components=rank,
                                      n_iter='auto',
                                      power_iteration_normalizer='QR',
                                      random_state=seed)

        sqrt_Sigma = np.diag(np.power(sigma, 1/2))

        V_star = Vt.T @ sqrt_Sigma

        Q = R @ V_star
        W = np.linalg.inv(Q.T @ Q + reg_weight * np.identity(rank)) @ Q.T @ R

        # instead of computing and storing the entire score matrix, just store Q and W and compute the scores on demand

        # torch doesn't support sparse tensor slicing, so will do everything with np/scipy
        self.user_embeddings = Q
        self.item_embeddings = W

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        return torch.from_numpy((self.user_embeddings[user, :] * self.item_embeddings[:, item].T).sum(axis=1))

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        r = self.user_embeddings[user, :] @ self.item_embeddings
        return torch.from_numpy(r.flatten())
