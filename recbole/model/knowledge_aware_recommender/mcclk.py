# -*- coding: utf-8 -*-
# @Time   : 2022/8/22
# @Author : Bowen Zheng
# @Email  : 18735382001@163.com

r"""
MCCLK
##################################################
Reference:
    Ding Zou et al. "Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System." in SIGIR 2022.

Reference code:
    https://github.com/CCIIPLab/MCCLK
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class Aggregator(nn.Module):
    def __init__(self, item_only=False, attention=True):
        super(Aggregator, self).__init__()

        # Only aggregate item embedding
        self.item_only = item_only
        # Whether use attention mechanism
        self.attention = attention

    def forward(
        self, entity_emb, user_emb, relation_emb, edge_index, edge_type, inter_matrix
    ):
        from torch_scatter import scatter_softmax, scatter_mean

        n_entities = entity_emb.shape[0]

        # KG aggregate
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type]
        neigh_relation_emb = (
            entity_emb[tail] * edge_relation_emb
        )  # [-1, embedding_size]

        if self.attention:
            # Calculate attention weights
            neigh_relation_emb_weight = self.calculate_sim_hrt(
                entity_emb[head], entity_emb[tail], edge_relation_emb
            )
            # [-1, 1] -> [-1, embedding_size]
            neigh_relation_emb_weight = neigh_relation_emb_weight.expand(
                neigh_relation_emb.shape[0], neigh_relation_emb.shape[1]
            )
            neigh_relation_emb_weight = scatter_softmax(
                neigh_relation_emb_weight, index=head, dim=0
            )  # [-1, embedding_size]
            neigh_relation_emb = torch.mul(
                neigh_relation_emb_weight, neigh_relation_emb
            )

        entity_agg = scatter_mean(
            src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0
        )  # [n_entities, embedding_size]

        # Only aggregate item embedding
        if self.item_only:
            return entity_agg

        user_agg = torch.sparse.mm(
            inter_matrix, entity_emb
        )  # [n_users, embedding_size]
        # The importance of relation to user
        score = torch.mm(user_emb, relation_emb.t())  # [n_users, n_relations]
        score = torch.softmax(score, dim=-1)
        user_agg = user_agg + (torch.mm(score, relation_emb)) * user_agg

        return entity_agg, user_agg

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):
        r"""
        The calculation method of attention weight here follows the code implementation of the author, which is
        slightly different from that described in the paper.
        """
        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        # [-1, 1, embedding_size] * [-1, embedding_size, 1] -> [-1, 1]
        att_weights = torch.matmul(
            head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)
        ).squeeze(dim=-1)
        att_weights = att_weights**2
        return att_weights


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(
        self,
        config,
        embedding_size,
        n_relations,
        edge_index,
        edge_type,
        inter_matrix,
        device,
    ):
        super(GraphConv, self).__init__()

        # load parameters info
        self.n_relations = n_relations
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.inter_matrix = inter_matrix
        self.embedding_size = embedding_size
        self.n_hops = config["n_hops"]
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.topk = config["k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.build_graph_separately = config["build_graph_separately"]
        self.device = device

        # define layers
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        # User a separate GCN to build item-item graph
        if self.build_graph_separately:
            r"""
            In the original author's implementation(https://github.com/CCIIPLab/MCCLK), the process of constructing
            k-Nearest-Neighbor item-item semantic graph(section 4.1 in paper) and encoding structural view(section 4.3.1 in paper)
            are combined. This implementation improves the computational efficiency, but is slightly different from the
            model structure described in the paper. We use the parameter `build_graph_separately` to control whether to
            use a separate GCN to build a item-item semantic graph. If `build_graph_separately` is set to true, the model
            structure will be the same as that described in the paper. Otherwise, the author's code implementation will be followed.
            """
            self.bg_convs = nn.ModuleList()
            for i in range(self.n_hops):
                self.bg_convs.append(Aggregator(item_only=True, attention=False))

        self.convs = nn.ModuleList()
        for i in range(self.n_hops):
            self.convs.append(Aggregator())

        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)  # node dropout
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  # mess dropout

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return edge_index[:, random_indices], edge_type[random_indices]

    def forward(self, user_emb, entity_emb):
        # node dropout
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.edge_sampling(
                self.edge_index, self.edge_type, self.node_dropout_rate
            )
            inter_matrix = self.node_dropout(self.inter_matrix)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            inter_matrix = self.inter_matrix

        origin_entity_emb = entity_emb

        entity_res_emb = [entity_emb]  # [n_entities, embedding_size]
        user_res_emb = [user_emb]  # [n_users, embedding_size]
        relation_emb = self.relation_embedding.weight  # [n_relations, embedding_size]
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](
                entity_emb, user_emb, relation_emb, edge_index, edge_type, inter_matrix
            )
            # message dropout
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            # result embedding
            entity_res_emb.append(entity_emb)
            user_res_emb.append(user_emb)

        entity_res_emb = torch.stack(entity_res_emb, dim=1)
        entity_res_emb = entity_res_emb.mean(dim=1, keepdim=False)
        user_res_emb = torch.stack(user_res_emb, dim=1)
        user_res_emb = user_res_emb.mean(dim=1, keepdim=False)

        # build item-item graph
        if self.build_graph_separately:
            item_adj = self._build_graph_separately(origin_entity_emb)
        else:
            # build origin item-item graph
            origin_item_adj = self.build_adj(origin_entity_emb, self.topk)
            # update item-item graph
            item_adj = (1 - self.lambda_coeff) * self.build_adj(
                entity_res_emb, self.topk
            ) + self.lambda_coeff * origin_item_adj

        return entity_res_emb, user_res_emb, item_adj

    def build_adj(self, context, topk):
        r"""Construct a k-Nearest-Neighbor item-item semantic graph.

        Returns:
            Sparse tensor of the normalized item-item matrix.
        """

        # construct similarity adj matrix
        n_entities = context.shape[0]
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).cpu()
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        # knn_val: [n_entities, topk]  knn_index: [n_entities, topk]
        knn_val, knn_index = torch.topk(sim, topk, dim=-1)
        knn_val, knn_index = knn_val.to(self.device), knn_index.to(self.device)

        y = knn_index.reshape(-1)
        x = (
            torch.arange(0, n_entities).unsqueeze(dim=-1).to(self.device)
        )  # [n_entities, 1]
        x = x.expand(n_entities, topk).reshape(-1)
        indice = torch.cat(
            (x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0
        )  # [2, n_entities * topk]
        value = knn_val.reshape(-1)
        adj_sparsity = torch.sparse.FloatTensor(
            indice.data, value.data, torch.Size([n_entities, n_entities])
        ).to(self.device)

        # normalized laplacian adj
        rowsum = torch.sparse.sum(adj_sparsity, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt_value = d_inv_sqrt._values()
        x = torch.arange(0, n_entities).unsqueeze(dim=0).to(self.device)
        x = x.expand(2, n_entities)
        d_mat_inv_sqrt_indice = x
        d_mat_inv_sqrt = torch.sparse.FloatTensor(
            d_mat_inv_sqrt_indice,
            d_mat_inv_sqrt_value,
            torch.Size([n_entities, n_entities]),
        )
        L_norm = torch.sparse.mm(
            torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity), d_mat_inv_sqrt
        )
        return L_norm

    def _build_graph_separately(self, entity_emb):
        # node dropout
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.edge_sampling(
                self.edge_index, self.edge_type, self.node_dropout_rate
            )
            inter_matrix = self.node_dropout(self.inter_matrix)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            inter_matrix = self.inter_matrix

        origin_item_adj = self.build_adj(entity_emb, self.topk)

        entity_res_emb = [entity_emb]  # [n_entities, embedding_size]
        relation_emb = self.relation_embedding.weight  # [n_relations, embedding_size]
        for i in range(len(self.bg_convs)):
            entity_emb = self.bg_convs[i](
                entity_emb, None, relation_emb, edge_index, edge_type, inter_matrix
            )
            # message dropout
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)
            # result embedding
            entity_res_emb.append(entity_emb)

        entity_res_emb = torch.stack(entity_res_emb, dim=1)
        entity_res_emb = entity_res_emb.mean(dim=1, keepdim=False)

        item_adj = (1 - self.lambda_coeff) * self.build_adj(
            entity_res_emb, self.topk
        ) + self.lambda_coeff * origin_item_adj

        return item_adj


class MCCLK(KnowledgeRecommender):
    r"""MCCLK is a knowledge-based recommendation model.
    It focuses on the contrastive learning in KG-aware recommendation and proposes a novel multi-level cross-view
    contrastive learning mechanism. This model comprehensively considers three different graph views for KG-aware
    recommendation, including global-level structural view, local-level collaborative and semantic views. It hence
    performs contrastive learning across three views on both local and global levels, mining comprehensive graph
    feature and structure information in a self-supervised manner.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MCCLK, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.reg_weight = config["reg_weight"]
        self.lightgcn_layer = config["lightgcn_layer"]
        self.item_agg_layer = config["item_agg_layer"]
        self.temperature = config["temperature"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.loss_type = config["loss_type"]

        # load dataset info
        self.inter_matrix = dataset.inter_matrix(form="coo").astype(
            np.float32
        )  # [n_users, n_items]
        # inter_matrix: [n_users, n_entities]; inter_graph: [n_users + n_entities, n_users + n_entities]
        self.inter_matrix, self.inter_graph = self.get_norm_inter_matrix(mode="si")
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )  # [n_entities, n_entities]
        # edge_index: [2, -1]; edge_type: [-1,]
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)

        # define layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.gcn = GraphConv(
            config=config,
            embedding_size=self.embedding_size,
            n_relations=self.n_relations,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            inter_matrix=self.inter_matrix,
            device=self.device,
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
        )
        # define loss
        if self.loss_type.lower() == "bpr":
            self.rec_loss = BPRLoss()
        elif self.loss_type.lower() == "bce":
            self.sigmoid = nn.Sigmoid()
            self.rec_loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(
                f"The loss type [{self.loss_type}] has not been supported."
            )
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_norm_inter_matrix(self, mode="bi"):
        # Get the normalized interaction matrix of users and items.

        def _bi_norm_lap(A):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(A.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(A):
            # D^{-1}A
            rowsum = np.array(A.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(A)
            return norm_adj.tocoo()

        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_entities, self.n_users + self.n_entities),
            dtype=np.float32,
        )
        inter_M = self.inter_matrix
        inter_M_t = self.inter_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        if mode == "bi":
            L = _bi_norm_lap(A)
        elif mode == "si":
            L = _si_norm_lap(A)
        else:
            raise NotImplementedError(
                f"Normalize mode [{mode}] has not been implemented."
            )
        # covert norm_inter_graph to tensor
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        norm_graph = torch.sparse.FloatTensor(i, data, L.shape)

        # interaction: user->item, [n_users, n_entities]
        L_ = L.tocsr()[: self.n_users, self.n_users :].tocoo()
        # covert norm_inter_matrix to tensor
        i_ = torch.LongTensor(np.array([L_.row, L_.col]))
        data_ = torch.FloatTensor(L_.data)
        norm_matrix = torch.sparse.FloatTensor(i_, data_, L_.shape)

        return norm_matrix.to(self.device), norm_graph.to(self.device)

    def get_edges(self, graph):
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(self.device), type.to(self.device)

    def forward(self):
        user_emb = self.user_embedding.weight
        entity_emb = self.entity_embedding.weight
        # Construct a k-Nearest-Neighbor item-item semantic graph and Structural View Encoder
        entity_gcn_emb, user_gcn_emb, item_adj = self.gcn(user_emb, entity_emb)
        # Semantic View Encoder
        item_semantic_emb = [entity_emb]
        item_agg_emb = entity_emb
        for i in range(self.item_agg_layer):
            item_agg_emb = torch.sparse.mm(item_adj, item_agg_emb)
            item_semantic_emb.append(item_agg_emb)
        item_semantic_emb = torch.stack(item_semantic_emb, dim=1)
        item_semantic_emb = item_semantic_emb.mean(dim=1, keepdim=False)
        # item_semantic_emb = F.normalize(item_semantic_emb, p=2, dim=1)

        # Collaborative View Encoder
        user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(
            user_emb, entity_emb, self.inter_graph
        )

        return (
            item_semantic_emb,
            user_lightgcn_emb,
            item_lightgcn_emb,
            user_gcn_emb,
            entity_gcn_emb,
        )

    def light_gcn(self, user_embedding, item_embedding, adj):
        ego_embeddings = torch.cat((user_embedding, item_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_entities], dim=0
        )
        return u_g_embeddings, i_g_embeddings

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        all_item = torch.cat((pos_item, neg_item), dim=0)

        (
            item_semantic_emb,
            user_lightgcn_emb,
            item_lightgcn_emb,
            user_gcn_emb,
            item_gcn_emb,
        ) = self.forward()
        item_emb_1 = item_semantic_emb[all_item]
        user_emb_1 = user_lightgcn_emb[user]
        item_emb_2 = item_lightgcn_emb[all_item]
        user_emb_2 = user_gcn_emb[user]
        item_emb_3 = item_gcn_emb[all_item]

        local_loss = self.local_level_loss(item_emb_1, item_emb_2)
        global_loss = self.global_level_loss_1(
            user_emb_2, user_emb_1
        ) + self.global_level_loss_2(item_emb_3, item_emb_1 + item_emb_2)

        user_embedding = torch.cat((user_emb_2, user_emb_1), dim=-1)
        pos_item_embedding = torch.cat(
            (
                item_gcn_emb[pos_item],
                item_semantic_emb[pos_item] + item_lightgcn_emb[pos_item],
            ),
            dim=-1,
        )
        neg_item_embedding = torch.cat(
            (
                item_gcn_emb[neg_item],
                item_semantic_emb[neg_item] + item_lightgcn_emb[neg_item],
            ),
            dim=-1,
        )

        pos_scores = torch.mul(user_embedding, pos_item_embedding).sum(dim=1)
        neg_scores = torch.mul(user_embedding, neg_item_embedding).sum(dim=1)
        if self.loss_type.lower() == "bpr":
            rec_loss = self.rec_loss(pos_scores, neg_scores)
        else:
            predict = torch.cat((pos_scores, neg_scores))
            target = torch.zeros(len(pos_item) + len(neg_item), dtype=torch.float32).to(
                self.device
            )
            target[: len(pos_item)] = 1
            rec_loss = self.rec_loss(predict, target)

        reg_loss = self.reg_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        loss = (
            rec_loss
            + self.reg_weight * reg_loss
            + self.beta * (self.alpha * local_loss + (1 - self.alpha) * global_loss)
        )

        return loss

    def local_level_loss(self, A_embedding, B_embedding):
        # The loss of local-level contrastive learning
        f = lambda x: torch.exp(x / self.temperature)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))
        local_loss = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )
        local_loss = local_loss.mean()
        return local_loss

    def global_level_loss_1(self, A_embedding, B_embedding):
        # The user embedding loss of global-level contrastive learning
        f = lambda x: torch.exp(x / self.temperature)
        A_embedding = self.fc2(A_embedding)
        B_embedding = self.fc2(B_embedding)

        refl_sim_1 = f(self.sim(A_embedding, A_embedding))
        between_sim_1 = f(self.sim(A_embedding, B_embedding))
        loss_1 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag())
        )

        refl_sim_2 = f(self.sim(B_embedding, B_embedding))
        between_sim_2 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_2.diag()
            / (refl_sim_2.sum(1) + between_sim_2.sum(1) - refl_sim_2.diag())
        )

        global_user_loss = (loss_1 + loss_2) * 0.5
        global_user_loss = global_user_loss.mean()
        return global_user_loss

    def global_level_loss_2(self, A_embedding, B_embedding):
        # The item embedding loss of global-level contrastive learning
        f = lambda x: torch.exp(x / self.temperature)
        A_embedding = self.fc3(A_embedding)
        B_embedding = self.fc3(B_embedding)

        refl_sim_1 = f(self.sim(A_embedding, A_embedding))
        between_sim_1 = f(self.sim(A_embedding, B_embedding))
        loss_1 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag())
        )

        refl_sim_2 = f(self.sim(B_embedding, B_embedding))
        between_sim_2 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_2.diag()
            / (refl_sim_2.sum(1) + between_sim_2.sum(1) - refl_sim_2.diag())
        )

        global_item_loss = (loss_1 + loss_2) * 0.5
        global_item_loss = global_item_loss.mean()
        return global_item_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        (
            item_semantic_emb,
            user_lightgcn_emb,
            item_lightgcn_emb,
            user_gcn_emb,
            item_gcn_emb,
        ) = self.forward()
        item_emb_1 = item_semantic_emb[item]
        user_emb_1 = user_lightgcn_emb[user]
        item_emb_2 = item_lightgcn_emb[item]
        user_emb_2 = user_gcn_emb[user]
        item_emb_3 = item_gcn_emb[item]

        user_embedding = torch.cat((user_emb_2, user_emb_1), dim=-1)
        item_embedding = torch.cat((item_emb_3, item_emb_1 + item_emb_2), dim=-1)

        scores = torch.mul(user_embedding, item_embedding).sum(dim=1)
        if self.loss_type.lower() == "bce":
            scores = self.sigmoid(scores)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            (
                item_semantic_emb,
                user_lightgcn_emb,
                item_lightgcn_emb,
                user_gcn_emb,
                entity_gcn_emb,
            ) = self.forward()
            self.restore_user_e = torch.cat((user_gcn_emb, user_lightgcn_emb), dim=-1)
            self.restore_entity_e = torch.cat(
                (entity_gcn_emb, item_semantic_emb + item_lightgcn_emb), dim=-1
            )

        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[: self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        if self.loss_type.lower() == "bce":
            scores = self.sigmoid(scores)

        return scores.view(-1)
