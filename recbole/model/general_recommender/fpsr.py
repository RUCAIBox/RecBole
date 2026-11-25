# -*- coding: utf-8 -*-
# @Time   : 2023/2/21
# @Author : Tianjun Wei
# @Email  : tjwei2-c@my.cityu.edu.hk

r"""
FPSR
################################################
Reference:
    Tianjun Wei et al. "Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation." in WWW 2023.

Reference code:
    https://github.com/Joinn99/FPSR/
"""

import torch
import numpy as np

from recbole.utils import InputType
from recbole.utils.enum_type import ModelType
from recbole.model.abstract_recommender import GeneralRecommender


class FPSR(GeneralRecommender):
    r"""FPSR is an item-based model for collaborative filtering.

    FPSR introduces graph partitioning on the item-item adjacency graph, aiming to restrict the scale of item similarity modeling.
    Specifically, it shows that the spectral information of the original item-item adjacency graph is well in preserving global-level information.
    Then, the global-level information is added to fine-tune local item similarities with a new data augmentation strategy acted as partition-aware
    prior knowledge, jointly to cope with the information loss brought by partitioning.

    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.eigen_dim = config["eigenvectors"]
        self.lambda_ = config["lambda"]
        self.rho = config["rho"]
        self.theta_1 = config["theta_1"]
        self.theta_2 = config["theta_2"]
        self.eta = config["eta"]
        self.tol = config["tol"]
        self.tau = config["tau"]

        # dummy params for recbole
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        # load dataset info
        self.inter = dataset.inter_matrix(form="coo")  # user-item interaction matrix
        self.inter = (
            torch.sparse_coo_tensor(
                torch.LongTensor(np.array([self.inter.row, self.inter.col])),
                torch.FloatTensor(self.inter.data),
                size=self.inter.shape,
                dtype=torch.float,
            )
            .coalesce()
            .to(self.device)
        )

        # storage variables for item similarity matrix S
        self.S_indices = []
        self.S_values = []

        # training process
        self.update_W()
        # calaulate W and generate first split
        first_split = self.partitioning(self.V)
        self.update_S(
            torch.arange(self.n_items, device=self.device)[torch.where(first_split)[0]]
        )  # recursive paritioning #1
        self.update_S(
            torch.arange(self.n_items, device=self.device)[torch.where(~first_split)[0]]
        )  # recursive paritioning #2

        self.S = (
            torch.sparse_coo_tensor(
                indices=torch.cat(self.S_indices, dim=1),
                values=torch.cat(self.S_values, dim=0),
                size=(self.n_items, self.n_items),
            )
            .coalesce()
            .T.to_sparse_csr()
        )
        del self.S_indices, self.S_values

    def _degree(self, inter_mat=None, dim=0, exp=-0.5) -> torch.Tensor:
        r"""Get the degree of users and items.

        Returns:
            Tensor of the node degrees.
        """
        if inter_mat is None:
            inter_mat = self.inter
        d_inv = torch.nan_to_num(
            torch.sparse.sum(inter_mat, dim=dim).to_dense().pow(exp),
            nan=0,
            posinf=0,
            neginf=0,
        )
        return d_inv

    def _svd(self, mat, k) -> torch.Tensor:
        r"""Perform truncated singular value decomposition (SVD) on
        the input matrix, return top-k eigenvectors.

        Returns:
            Tok-k eigenvectors.
        """
        _, _, V = torch.svd_lowrank(mat, q=min(4 * k, mat.shape[-1]), niter=10)
        return V[:, :k]

    def _norm_adj(self, item_list=None) -> torch.Tensor:
        r"""Get the normalized item-item adjacency matrix for a group of items.

        Returns:
            Sparse tensor of the normalized item-item adjacency matrix.
        """
        if item_list is None:
            vals = self.inter.values() * self.d_i[self.inter.indices()[1]].squeeze()
            return torch.sparse_coo_tensor(
                self.inter.indices(),
                self._degree(dim=1)[self.inter.indices()[0]] * vals,
                size=self.inter.shape,
                dtype=torch.float,
            ).coalesce()
        else:
            inter = self.inter.index_select(dim=1, index=item_list).coalesce()
            vals = inter.values() * self.d_i[item_list][inter.indices()[1]].squeeze()
            return torch.sparse_coo_tensor(
                inter.indices(),
                self._degree(inter, dim=1)[inter.indices()[0]] * vals,
                size=inter.shape,
                dtype=torch.float,
            ).coalesce()

    def update_W(self) -> None:
        r"""Derive the global-level information matrix W, and use the eigenvector
        with the second largest eigenvalue to perform first graph partitioning.

        """
        self.d_i = self._degree(dim=0).reshape(-1, 1)
        self.d_i_inv = self._degree(dim=0, exp=0.5).reshape(1, -1)
        self.V = self._svd(self._norm_adj(), self.eigen_dim)

    def partitioning(self, V) -> torch.Tensor:
        r"""Perform graph bipartitioning.

        Returns:
            Paritioning result.
        """
        split = V[:, 1] >= 0
        if split.sum() == split.shape[0] or split.sum() == 0:
            split = V[:, 1] >= torch.median(V[:, 1])
        return split

    def update_S(self, item_list) -> None:
        r"""Derive partition-aware item similarity matrix S in each partition."""
        if item_list.shape[0] <= self.tau * self.n_items:
            # If the partition size is samller than size limit, model item similarity for this partition.
            comm_inter = self.inter.index_select(dim=1, index=item_list).to_dense()
            comm_inter = torch.mm(comm_inter.T, comm_inter)
            comm_ae = self.item_similarity(
                comm_inter,
                self.V[item_list, :],
                self.d_i[item_list, :],
                self.d_i_inv[:, item_list],
            )
            comm_ae = torch.where(comm_ae >= self.tol, comm_ae, 0).to_sparse_coo()
            self.S_indices.append(item_list[comm_ae.indices()])
            self.S_values.append(comm_ae.values())
        else:
            # If the partition size is larger than size limit, perform graph partitioning on this partition.
            split = self.partitioning(self._svd(self._norm_adj(item_list), 2))
            self.update_S(item_list[torch.where(split)[0]])
            self.update_S(item_list[torch.where(~split)[0]])

    def item_similarity(self, inter_mat, V, d_i, d_i_inv) -> torch.Tensor:
        r"""Update partition-aware item similarity matrix S in a specific partition.

        Returns:
            Partition-aware item similarity matrix of a partition.
        """
        # Initialize
        Q_hat = (
            inter_mat
            + self.theta_2 * torch.diag(torch.pow(d_i_inv.squeeze(), 2))
            + self.eta
        )
        Q_inv = torch.inverse(
            Q_hat + self.rho * torch.eye(inter_mat.shape[0], device=self.device)
        )
        Z_aux = (
            Q_inv
            @ Q_hat
            @ (
                torch.eye(inter_mat.shape[0], device=self.device)
                - self.lambda_ * d_i * V @ V.T * d_i_inv
            )
        )
        del Q_hat
        Phi = torch.zeros_like(Q_inv, device=self.device)
        S = torch.zeros_like(Q_inv, device=self.device)
        for _ in range(50):
            # Iteration
            Z_tilde = Z_aux + Q_inv @ (self.rho * (S - Phi))
            gamma = torch.diag(Z_tilde) / (torch.diag(Q_inv) + 1e-10)
            Z = Z_tilde - Q_inv * gamma  # update Z
            S = torch.clip(Z + Phi - self.theta_1 / self.rho, min=0)  # update S
            Phi += Z - S  # update Phi
        return S

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = self.inter.index_select(
            dim=0, index=interaction[self.USER_ID]
        ).to_dense()
        item = self.S.index_select(dim=1, index=interaction[self.ITEM_ID]).to_dense()
        d_i_inv = self.d_i_inv[:, interaction[self.ITEM_ID]]
        V = self.V[interaction[self.ITEM_ID], :]

        r = torch.mul(item.T, user).sum(dim=-1)
        r += self.lambda_ * torch.mul(user * self.d_i.T @ self.V, V * d_i_inv.T).sum(
            dim=-1
        )
        return r

    def full_sort_predict(self, interaction):
        user = self.inter.index_select(
            dim=0, index=interaction[self.USER_ID]
        ).to_dense()
        r = torch.sparse.mm(self.S, user.T).T
        r += self.lambda_ * user * self.d_i.T @ self.V @ self.V.T * self.d_i_inv
        return r
