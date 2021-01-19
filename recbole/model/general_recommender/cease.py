
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


def encode_categorical_item_features(dataset, included_features):
    item_features = dataset.get_item_feature()

    mlb = MultiLabelBinarizer(sparse_output=True)
    ohe = OneHotEncoder(sparse=True)

    encoded_feats = []

    for feat in included_features:
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
                f'CEASE/A-EASE only supports token or token_seq types. [{feat}] is of type [{t}].')

    if not encoded_feats:
        raise ValueError(
            f'No valid token or token_seq features to include.')

    return sp.hstack(encoded_feats).T.astype(np.float32)


class CEASE(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        B = dataset.inter_matrix(
            form='csr').astype(np.float32)

        item_feat_weight = config['item_feat_weight']
        reg_weight = config['reg_weight']
        included_features = config['included_features']

        T = encode_categorical_item_features(dataset, included_features)
        T *= item_feat_weight

        # just directly calculate the entire score matrix in init
        # (can't be done incrementally)

        X = sp.vstack([B, T]).tocsr()

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
        self.interaction_tag_matrix = X

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        return torch.from_numpy((self.interaction_tag_matrix[user, :].multiply(self.item_similarity[:, item].T)).sum(axis=1).getA1())

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        r = self.interaction_tag_matrix[user, :] @ self.item_similarity
        return torch.from_numpy(r.flatten())
