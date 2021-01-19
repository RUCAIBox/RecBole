
r"""
C-EASE
################################################
Reference:
    Olivier Jeunen, et al. "Closed-Form Models for Collaborative Filtering with Side-Information".

Reference code:
    https://github.com/olivierjeunen/ease-side-info-recsys-2020/
"""


from recbole.utils.enum_type import ModelType, FeatureType
import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def encode_categorical_item_features(dataset, selected_features):
    item_features = dataset.get_item_feature()

    mlb = MultiLabelBinarizer(sparse_output=True)
    ohe = OneHotEncoder(sparse=True)

    encoded_feats = []

    for feat in selected_features:
        t = dataset.field2type[feat]
        feat_frame = item_features[feat].numpy()

        if t == FeatureType.TOKEN:
            encoded = ohe.fit_transform(feat_frame.reshape(-1, 1))
            encoded_feats.append(encoded)
        elif t == FeatureType.TOKEN_SEQ:
            encoded = mlb.fit_transform(feat_frame)

            # drop first column which corresponds to the padding 0; real categories start at 1
            # convert to csc first?
            encoded = encoded[:, 1:]
            encoded_feats.append(encoded)
        else:
            raise Warning(
                f'CEASE only supports token or token_seq types. [{feat}] is of type [{t}].')

    if not encoded_feats:
        raise ValueError(
            f'No valid token or token_seq features to include.')

    return sp.hstack(encoded_feats).T.astype(np.float32)


def ease_like(M, reg_weight):
    # gram matrix
    G = M.T @ M

    # add reg to diagonal
    G += reg_weight * sp.identity(G.shape[0])

    # convert to dense because inverse will be dense
    G = G.todense()

    # invert. this takes most of the time
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    # zero out diag
    np.fill_diagonal(B, 0.)

    return B


class CEASE(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        inter_matrix = dataset.inter_matrix(
            form='csr').astype(np.float32)

        item_feat_weight = config['item_feat_weight']
        reg_weight = config['reg_weight']
        selected_features = config['selected_features']

        tag_item_matrix = item_feat_weight * \
            encode_categorical_item_features(dataset, selected_features)

        # just directly calculate the entire score matrix in init
        # (can't be done incrementally)

        X = sp.vstack([inter_matrix, tag_item_matrix]).tocsr()

        item_similarity = ease_like(X, reg_weight)

        # instead of computing and storing the entire score matrix, just store B and compute the scores on demand
        # more memory efficient for a larger number of users

        # torch doesn't support sparse tensor slicing, so will do everything with np/scipy
        self.item_similarity = item_similarity
        self.interaction_matrix = inter_matrix

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
