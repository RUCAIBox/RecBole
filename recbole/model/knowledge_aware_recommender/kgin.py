# -*- coding: utf-8 -*-
# @Time   : 2021/3/25
# @Author : Wenqi Sun
# @Email  : wenqisun@pku.edu.cn

r"""
KGIN
##################################################
Reference:
    Xiang Wang et al. "Learning Intents behind Interactions with Knowledge Graph for Recommendation." in WWW 2021.

Reference code:
    https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm
from torch_scatter import scatter_mean
from collections import defaultdict

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization, xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_factors):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors

    def forward(
        self, entity_emb, user_emb, latent_embedding, edge_index, edge_type, interact_mat, weight, disen_weight_att
    ):
        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors
        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_embedding.weight.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]
        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att), weight).expand(n_users, n_factors, channel)
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(
        self,
        channel,
        n_hops,
        n_users,
        n_factors,
        n_relations,
        interact_mat,
        ind,
        tmp,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1
    ):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = tmp

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _cul_cor(self):

        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = -torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(
        self,
        user_emb,
        entity_emb,
        latent_embedding,
        edge_index,
        edge_type,
        interact_mat,
        mess_dropout=True,
        node_dropout=False
    ):
        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](
                entity_emb, user_emb, latent_embedding, edge_index, edge_type, interact_mat, self.weight,
                self.disen_weight_att
            )
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor


class KGIN(KnowledgeRecommender):
    r"""KGIN is a knowledge-aware recommendation model. It combines knowledge graph and the user-item interaction
    graph to a new graph called collaborative knowledge graph (CKG). This model explores intents behind a user-item
    interaction by using auxiliary item knowledge.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGIN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_factors = config['n_factors']
        self.context_hops = config['context_hops']
        self.decay = config['l2']
        self.sim_decay = config['sim_regularity']
        self.node_dropout = config['node_dropout']
        self.node_dropout_rate = config['node_dropout_rate']
        self.mess_dropout = config['mess_dropout']
        self.mess_dropout_rate = config['mess_dropout_rate']
        self.ind = config['ind']
        self.reg_weight = config['reg_weight']
        self.temperature = config['temperature']

        # define layers and loss
        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.latent_embedding = nn.Embedding(self.n_factors, self.embedding_size)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # load dataset info
        self.kg_graph = dataset.kg_graph(form='coo', value_field='relation_id')
        self.triplets = zip(self.kg_graph.row, self.kg_graph.data, self.kg_graph.col)
        self.interaction_feat = dataset.dataset.inter_feat
        self.interactions = zip(
            self.interaction_feat[self.USER_ID].numpy(), self.interaction_feat[self.ITEM_ID].numpy()
        )
        self.graph, self.relation_dict = self.init_graph(self.interactions, self.triplets)
        self.edge_index, self.edge_type = self._get_edges(self.graph)
        self.adj_mat_list, self.mean_mat_list = self.init_sparse_relational_graph(self.relation_dict)
        self.adj_mat = self.mean_mat_list[0]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.gcn = self.init_model()

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def init_graph(self, interactions, triplets):
        graph = nx.MultiDiGraph()
        rd = defaultdict(list)

        for u_id, i_id in tqdm(interactions, ascii=True):
            rd[0].append([u_id, i_id])

        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            graph.add_edge(h_id, t_id, key=r_id)
            rd[r_id].append([h_id, t_id])

        return graph, rd

    def init_sparse_relational_graph(self, relation_dict):

        def _si_norm_lap(adj):
            # D^{-1}A
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        adj_mat_list = []
        for r_id in tqdm(relation_dict.keys()):
            np_mat = np.array(relation_dict[r_id])
            if r_id == 0:
                cf = np_mat.copy()
                cf[:, 1] = cf[:, 1] + self.n_users  # [0, n_items) -> [n_users, n_users+n_items)
                vals = [1.] * len(cf)
                adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(self.n_nodes, self.n_nodes))
            else:
                vals = [1.] * len(np_mat)
                adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(self.n_nodes, self.n_nodes))
            adj_mat_list.append(adj)

        mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
        # interaction: user->item, [n_users, n_entities]
        mean_mat_list[0] = mean_mat_list[0].tocsr()[:self.n_users, self.n_users:].tocoo()

        return adj_mat_list, mean_mat_list

    def init_model(self):
        return GraphConv(
            channel=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            n_factors=self.n_factors,
            interact_mat=self.interact_mat,
            ind=self.ind,
            tmp=self.temperature,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate
        )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(
            user_embeddings,
            entity_embeddings,
            self.latent_embedding,
            self.edge_index,
            self.edge_type,
            self.interact_mat,
            mess_dropout=self.mess_dropout,
            node_dropout=self.node_dropout
        )

        return user_gcn_emb, entity_gcn_emb, cor

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings, cor = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        cor_loss = self.sim_decay * cor
        loss = mf_loss + self.reg_weight * reg_loss + cor_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings, cor = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e, cor = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[:self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
