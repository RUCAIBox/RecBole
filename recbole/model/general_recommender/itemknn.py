# -*- coding: utf-8 -*-
# @Time   : 2020/8/18
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com

r"""
ItemKNN
################################################
Reference:
    Aiolli,F et al. Efficient top-n recommendation for very large scale binary rated datasets.
    In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
"""

import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType, ModelType
from recbole.model.abstract_recommender import GeneralRecommender


class ComputeSimilarity:

    def __init__(self, dataMatrix, topk=100, shrink=0, normalize=True):
        r"""Computes the cosine similarity on the columns of dataMatrix

        If it is computed on :math:`URM=|users| \times |items|`, pass the URM.

        If it is computed on :math:`ICM=|items| \times |features|`, pass the ICM transposed.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink(int) :  hyper-parameter in calculate cosine distance.
            normalize (bool):   If True divide the dot product by the product of the norms.
        """

        super(ComputeSimilarity, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topk, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

    def compute_similarity(self, block_size=100):
        r"""Compute the similarity for the given dataset

        Args:
            block_size(int): divide matrix to :math:`n\_columns \div block\_size` to calculate cosine_distance

        Returns:
            scipy.sparse.csr_matrix: sparse matrix W shape of (self.n_columns, self.n_columns)
        """

        values = []
        rows = []
        cols = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
        sumOfSquared = np.sqrt(sumOfSquared)

        end_col_local = self.n_columns
        start_col_block = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            if item_data.ndim == 1:
                item_data = np.expand_dims(item_data, axis=1)

            # Compute item similarities

            this_block_weights = self.dataMatrix.T.dot(item_data)
            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights.squeeze()
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]

                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)

            start_col_block += block_size

        # End while on columns

        W_sparse = sp.csr_matrix((values, (rows, cols)),
                                 shape=(self.n_columns, self.n_columns),
                                 dtype=np.float32)
        return W_sparse.tocsc()


class ItemKNN(GeneralRecommender):

    r"""ItemKNN is a basic model that compute item similarity with the interaction matrix.

    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(ItemKNN, self).__init__(config, dataset)

        # load parameters info
        self.k = config['k']
        self.shrink = config['shrink'] if 'shrink' in config else 0.0

        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        self.w = ComputeSimilarity(self.interaction_matrix, topk=self.k, shrink=self.shrink).compute_similarity()
        self.pred_mat = self.interaction_matrix.dot(self.w).tolil()

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)
        result = []

        for index in range(len(user)):
            uid = user[index]
            iid = item[index]
            score = self.pred_mat[uid, iid]
            result.append(score)
        result = torch.from_numpy(np.array(result)).to(self.device)
        return result

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()

        score = self.pred_mat[user, :].toarray().flatten()
        result = torch.from_numpy(score).to(self.device)

        return result
