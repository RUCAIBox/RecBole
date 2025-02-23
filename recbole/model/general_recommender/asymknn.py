import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from scipy.sparse import csr_matrix


class ComputeSimilarity:
    def __init__(self, dataMatrix, topk=100, alpha=0.5, method="item"):
        r"""Computes the asymmetric cosine similarity of dataMatrix with alpha parameter.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) :    The k value in KNN.
            alpha (float):  Asymmetry control parameter in cosine similarity calculation.
            method (str) :  Caculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
        """

        super(ComputeSimilarity, self).__init__()

        self.method = method
        self.alpha = alpha

        self.n_rows, self.n_columns = dataMatrix.shape

        if self.method == "user":
            self.TopK = min(topk, self.n_rows)
        else:
            self.TopK = min(topk, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

    def compute_asym_similarity(self, block_size=100):
        r"""Compute the asymmetric cosine similarity for the given dataset.

        Args:
            block_size (int): Divide matrix into blocks for efficient calculation.

        Returns:
            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num],
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        if self.method == "user":
            sumOfMatrix = np.array(self.dataMatrix.sum(axis=1)).ravel()
            end_local = self.n_rows
        elif self.method == "item":
            sumOfMatrix = np.array(self.dataMatrix.sum(axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError("Make sure 'method' is in ['user', 'item']!")

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:
            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user or item
            if self.method == "user":
                data = self.dataMatrix[start_block:end_block, :]
            else:
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray()

            # Compute similarities
            if self.method == "user":
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)

            for index_in_block in range(this_block_size):
                this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0

                # Apply asymmetric cosine normalization
                denominator = (sumOfMatrix[Index] ** self.alpha) * (
                    sumOfMatrix ** (1 - self.alpha)
                ) + 1e-6
                this_line_weights = np.multiply(this_line_weights, 1 / denominator)

                # Sort indices and select TopK
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[
                    0 : self.TopK
                ]
                relevant_partition_sorting = np.argsort(
                    -this_line_weights[relevant_partition]
                )
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_line_weights[top_k_idx][notZerosMask])
                if self.method == "user":
                    rows.extend(np.ones(numNotZeros) * Index)
                    cols.extend(top_k_idx[notZerosMask])
                else:
                    rows.extend(top_k_idx[notZerosMask])
                    cols.extend(np.ones(numNotZeros) * Index)

            start_block += block_size

        # End while
        if self.method == "user":
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)),
                shape=(self.n_rows, self.n_rows),
                dtype=np.float32,
            )
        else:
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)),
                shape=(self.n_columns, self.n_columns),
                dtype=np.float32,
            )
        return neigh, W_sparse.tocsc()


class AsymKNN(GeneralRecommender):
    r"""AsymKNN: A traditional recommender model based on asymmetric cosine similarity and score prediction.

    AsymKNN computes user-item recommendations by leveraging asymmetric cosine similarity
    over the interaction matrix. This model allows for flexible adjustment of similarity
    calculations and scoring normalization via several tunable parameters.

    Config:
        k (int): Number of neighbors to consider in the similarity calculation.
        method (str): Specifies whether to calculate similarities based on users or items.
            Valid options are 'user' or 'item'.
        alpha (float): Weight parameter for asymmetric cosine similarity, controlling
            the importance of the interaction matrix in the similarity computation.
            Must be in the range [0, 1].
        q (int): Exponent for adjusting the 'locality of scoring function' after similarity computation.
        beta (float): Parameter for controlling the balance between factors in the
            final score normalization. Must be in the range [0, 1].

    Reference:
        Aiolli,F et al. Efficient top-n recommendation for very large scale binary rated datasets.
        In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(AsymKNN, self).__init__(config, dataset)

        # load parameters info
        self.k = config["k"]  # Size of neighborhood for cosine
        self.method = config[
            "knn_method"
        ]  # Caculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
        self.alpha = (
            config["alpha"] if "alpha" in config else 0.5
        )  # Asymmetric cosine parameter
        self.q = config["q"] if "q" in config else 1.0  # Weight adjustment exponent
        self.beta = (
            config["beta"] if "beta" in config else 0.5
        )  # Beta for final score normalization

        assert (
            0 <= self.alpha <= 1
        ), f"The asymmetric parameter 'alpha' must be value between in [0,1], but got {self.alpha}"
        assert (
            0 <= self.beta <= 1
        ), f"The asymmetric parameter 'beta' must be value between [0,1], but got {self.beta}"
        assert isinstance(
            self.k, int
        ), f"The neighborhood parameter 'k' must be an integer, but got {self.k}"
        assert isinstance(
            self.q, int
        ), f"The exponent parameter 'q' must be an integer, but got {self.q}"

        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        _, self.w = ComputeSimilarity(
            self.interaction_matrix, topk=self.k, alpha=self.alpha, method=self.method
        ).compute_asym_similarity()

        if self.method == "user":
            nominator = self.w.dot(self.interaction_matrix)
            factor1 = np.power(np.sqrt(self.w.power(2).sum(axis=1)), 2 * self.beta)
            factor2 = np.power(
                np.sqrt(self.interaction_matrix.power(2).sum(axis=0)),
                2 * (1 - self.beta),
            )
            denominator = factor1.dot(factor2) + 1e-6
        else:
            nominator = self.interaction_matrix.dot(self.w)
            factor1 = np.power(
                np.sqrt(self.interaction_matrix.power(2).sum(axis=1)), 2 * self.beta
            )
            factor2 = np.power(
                np.sqrt(self.w.power(2).sum(axis=1)), 2 * (1 - self.beta)
            )
            denominator = factor1.dot(factor2.T) + 1e-6

        self.pred_mat = csr_matrix(nominator / denominator).tolil()

        # Apply 'locality of scoring function' via q: f(w) = w^q
        self.pred_mat = self.pred_mat.power(self.q)

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["w", "pred_mat"]

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
