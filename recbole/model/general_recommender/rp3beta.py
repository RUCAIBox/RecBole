r"""
RP3Beta
################################################
Reference:
    Paudel, Bibek et al. Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications. https://doi.org/10.1145/2955101

Reference code:
    https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/master/GraphBased/RP3betaRecommender.py
"""


from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


def get_inv_degree_matrix(A):
    # add epsilon to degree sums to suppress warning about division by zero when we later divide
    degree_sums = A.sum(axis=1).getA1() + 1e-7

    return sp.diags(1/degree_sums)


# for reference, doing it in one computation
# since the resultant matrix is dense, I'll refrain from doing this
def calculate_rp3beta(B, beta):
    user_degree_inv = get_inv_degree_matrix(B)
    item_degree_inv = get_inv_degree_matrix(B.T)

    # multiplication on left for row-wise scaling
    user_transition = user_degree_inv @ B
    item_transition = item_degree_inv @ B.T

    P3 = user_transition @ item_transition @ user_transition

    # multiplication on right for column-wise scaling (i.e., we're reweighting by inverse item popularity to a power)
    RP3Beta = P3 @ item_degree_inv.power(beta)

    return RP3Beta


class RP3Beta(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        B = dataset.inter_matrix(
            form='coo').astype(np.float32)

        self.beta = config['beta']

        user_degree_inv = get_inv_degree_matrix(B)
        item_degree_inv = get_inv_degree_matrix(B.T)

        self.user_transition = user_degree_inv @ B
        self.item_transition = item_degree_inv @ B.T
        self.item_degree_inv = item_degree_inv

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        specific_user_transitions = self.user_transition[user]

        # make all item predictions for specified users
        user_all_items = specific_user_transitions @ self.item_transition @ self.user_transition @ self.item_degree_inv.power(
            self.beta)

        # then narrow down to specific items
        # without this copy(): "cannot set WRITEABLE flag..."
        item_predictions = user_all_items[range(len(user)), item.copy()]

        return torch.from_numpy(item_predictions.getA1())

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        specific_user_transitions = self.user_transition[user]

        item_predictions = specific_user_transitions @ self.item_transition @ self.user_transition @ self.item_degree_inv.power(
            self.beta)

        return torch.from_numpy(item_predictions.todense().getA1())
