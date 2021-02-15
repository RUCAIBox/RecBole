# -*- coding: utf-8 -*-
# @Time   : 2021/1/14
# @Author : Chengyuan Li
# @Email  : 2017202049@ruc.edu.cn

import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType

import numpy as np
import networkx as nx
import community
from sklearn.metrics import jaccard_score


class NNCF(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NNCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(
            np.float32)

        # load parameters info
        self.ui_embedding_size = config['ui_embedding_size']
        self.neigh_embedding_size = config['neigh_embedding_size']
        self.num_conv_kernel = config['num_conv_kernel']
        self.conv_kernel_size = config['conv_kernel_size']
        self.pool_kernel_size = config['pool_kernel_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.neigh_num = config['neigh_num']
        self.use_random = config['use_random']
        self.use_knn = config['use_knn']
        self.use_louvain = config['use_louvain']
        self.resolution = config['resolution']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.ui_embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.ui_embedding_size)
        self.user_neigh_embedding = nn.Embedding(self.n_items, self.neigh_embedding_size)
        self.item_neigh_embedding = nn.Embedding(self.n_users, self.neigh_embedding_size)
        self.user_conv = nn.Sequential(
            nn.Conv1d(self.neigh_embedding_size, self.num_conv_kernel, self.conv_kernel_size),
            nn.MaxPool1d(self.pool_kernel_size), 
            nn.ReLU())
        self.item_conv = nn.Sequential(
            nn.Conv1d(self.neigh_embedding_size, self.num_conv_kernel, self.conv_kernel_size),
            nn.MaxPool1d(self.pool_kernel_size), 
            nn.ReLU())
        conved_size = self.neigh_num - (self.conv_kernel_size - 1)
        pooled_size = (conved_size - (self.pool_kernel_size - 1) - 1) // self.pool_kernel_size + 1
        self.mlp_layers = MLPLayers([2 * pooled_size * self.num_conv_kernel + self.ui_embedding_size] + self.mlp_hidden_size, config['dropout'])
        self.sigmoid = nn.Sigmoid()
        self.out_layer = nn.Sequential(nn.Linear(self.mlp_hidden_size[-1], 1),
                                       nn.Sigmoid())
        self.dropout_layer = torch.nn.Dropout(p=config['dropout'])
        self.loss = nn.BCELoss()

        # choose the method to use neighborhood information
        if self.use_random:
            self.u_neigh, self.i_neigh = self.get_neigh_random()
        elif self.use_knn:
            self.u_neigh, self.i_neigh = self.get_neigh_knn()
        elif self.use_louvain:
            self.u_neigh, self.i_neigh = self.get_neigh_louvain()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    # Unify embedding length
    def Max_ner(self, lst, max_ner):
        for i in range(len(lst)):
            if len(lst[i]) >= max_ner:
                lst[i] = lst[i][:max_ner]
            else:
                length = len(lst[i])
                for _ in range(max_ner - length):
                    lst[i].append(0)
        return lst

    # Find other nodes in the same community
    def get_community_member(self, partition, community_dict, node, kind):
        comm = community_dict[partition[node]]
        return [x for x in comm if x.startswith(kind)]

    # Prepare neiborhood embeddings, i.e. I(u) and U(i)
    def prepare_vector_element(self, partition, relation, community_dict):
        item2user_neighbor_lst = [[] for _ in range(self.n_items)]  
        user2item_neighbor_lst = [[] for _ in range(self.n_users)]  

        for r in range(len(relation)):
            user, item = relation[r][0], relation[r][1]
            item2user_neighbor = self.get_community_member(partition, community_dict, user, 'u')
            np.random.shuffle(item2user_neighbor)
            user2item_neighbor = self.get_community_member(partition, community_dict, item, 'i')
            np.random.shuffle(user2item_neighbor)
            _, user = user.split('_', 1)
            user = int(user)
            _, item = item.split('_', 1)
            item = int(item)
            for i in range(len(item2user_neighbor)):
                name, index = item2user_neighbor[i].split('_', 1)
                item2user_neighbor[i] = int(index)
            for i in range(len(user2item_neighbor)):
                name, index = user2item_neighbor[i].split('_', 1)
                user2item_neighbor[i] = int(index)

            item2user_neighbor_lst[item] = item2user_neighbor
            user2item_neighbor_lst[user] = user2item_neighbor

        return user2item_neighbor_lst, item2user_neighbor_lst

    # Get neighborhood embeddings using louvain method
    def get_neigh_louvain(self):
        inter_M = self.interaction_matrix
        pairs = list(zip(inter_M.row, inter_M.col))

        tmp_relation = []
        for i in range(len(pairs)):
            tmp_relation.append(['user_' + str(pairs[i][0]), 'item_' + str(pairs[i][1])])

        G = nx.Graph()
        G.add_edges_from(tmp_relation)
        resolution = self.resolution
        partition = community.best_partition(G, resolution=resolution)

        community_dict = {}
        community_dict.setdefault(0, [])
        for i in range(len(partition.values())):
            community_dict[i] = []
        for node, part in partition.items():
            community_dict[part] = community_dict[part] + [node]

        tmp_user2item, tmp_item2user = self.prepare_vector_element(partition, tmp_relation, community_dict)
        u_neigh = self.Max_ner(tmp_user2item, self.neigh_num)
        i_neigh = self.Max_ner(tmp_item2user, self.neigh_num)

        u_neigh = torch.tensor(u_neigh, device=self.device)
        i_neigh = torch.tensor(i_neigh, device=self.device)
        return u_neigh, i_neigh

    # Count the similarity of node and direct neighbors using jaccard method
    def count_jaccard(self, inters, node, neigh_list, kind):
        if kind == 'u':
            if node in neigh_list:
                return 0
            vec_node = inters[:, node]
            score = 0
            for neigh in neigh_list:
                vec_neigh = inters[:, neigh]
                tmp = jaccard_score(vec_node, vec_neigh)
                score += tmp
            return score
        else:
            if node in neigh_list:
                return 0
            vec_node = inters[node]
            score = 0
            for neigh in neigh_list:
                vec_neigh = inters[neigh]
                tmp = jaccard_score(vec_node, vec_neigh)
                score += tmp
            return score

    # Get neighborhood embeddings using knn method
    def get_neigh_knn(self):
        inter_M = self.interaction_matrix
        pairs = list(zip(inter_M.row, inter_M.col))
        ui_inters = np.zeros((self.n_users, self.n_items), dtype=np.int8)

        for i in range(len(pairs)):
            ui_inters[pairs[i][0], pairs[i][1]] = 1

        u_neigh, i_neigh = [], []

        for u in range(self.n_users):
            neigh_list = ui_inters[u].nonzero()[0]
            direct_neigh_num = len(neigh_list)
            if len(neigh_list) == 0:
                u_neigh.append(self.neigh_num * [0])
            elif direct_neigh_num < self.neigh_num:
                tmp_k = self.neigh_num - direct_neigh_num
                knn_neigh_dict = {}
                for i in range(self.n_items):
                    score = self.count_jaccard(ui_inters, i, neigh_list, 'u')
                    knn_neigh_dict[i] = score
                knn_neigh_dict_sorted = dict(sorted(knn_neigh_dict.items(), key=lambda item:item[1], reverse=True))
                knn_neigh_list = knn_neigh_dict_sorted.keys()
                neigh_list = list(neigh_list) + list(knn_neigh_list)
                u_neigh.append(neigh_list[:self.neigh_num])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                u_neigh.append(neigh_list[mask])

        for i in range(self.n_items):
            neigh_list = ui_inters[:, i].nonzero()[0]
            direct_neigh_num = len(neigh_list)
            if len(neigh_list) == 0:
                i_neigh.append(self.neigh_num * [0])
            elif direct_neigh_num < self.neigh_num:
                tmp_k = self.neigh_num - direct_neigh_num
                knn_neigh_dict = {}
                for i in range(self.n_users):
                    score = self.count_jaccard(ui_inters, i, neigh_list, 'i')
                    knn_neigh_dict[i] = score
                knn_neigh_dict_sorted = dict(sorted(knn_neigh_dict.items(), key=lambda item:item[1], reverse=True))
                knn_neigh_list = knn_neigh_dict_sorted.keys()
                neigh_list = list(neigh_list) + list(knn_neigh_list)
                i_neigh.append(neigh_list[:self.neigh_num])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                i_neigh.append(neigh_list[mask])

        u_neigh = torch.tensor(u_neigh, device=self.device)
        i_neigh = torch.tensor(i_neigh, device=self.device)
        return u_neigh, i_neigh

    # Get neighborhood embeddings using random method
    def get_neigh_random(self):
        inter_M = self.interaction_matrix
        pairs = list(zip(inter_M.row, inter_M.col))
        ui_inters = np.zeros((self.n_users, self.n_items), dtype=np.int8)

        for i in range(len(pairs)):
            ui_inters[pairs[i][0], pairs[i][1]] = 1

        u_neigh, i_neigh = [], []

        for u in range(self.n_users):
            neigh_list = ui_inters[u].nonzero()[0]
            if len(neigh_list) == 0:
                u_neigh.append(self.neigh_num * [0])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                u_neigh.append(neigh_list[mask])

        for i in range(self.n_items):
            neigh_list = ui_inters[:, i].nonzero()[0]
            if len(neigh_list) == 0:
                i_neigh.append(self.neigh_num * [0])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                i_neigh.append(neigh_list[mask])

        u_neigh = torch.tensor(u_neigh, device=self.device)
        i_neigh = torch.tensor(i_neigh, device=self.device)
        return u_neigh, i_neigh

    # Get neighborhood embeddings
    def get_neigh_info(self, user, item):
        batch = user.size(0)
        batch_u_neigh = self.u_neigh[user]
        batch_i_neigh = self.i_neigh[item]
        return batch_u_neigh, batch_i_neigh

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)

        user_neigh_input, item_neigh_input = self.get_neigh_info(user, item)
        user_neigh_embedding = self.user_neigh_embedding(user_neigh_input)
        item_neigh_embedding = self.item_neigh_embedding(item_neigh_input)
        user_neigh_embedding = user_neigh_embedding.permute(0, 2, 1)
        user_neigh_conv_embedding = self.user_conv(user_neigh_embedding)
        # batch_size * out_channel * pool_size
        batch_size = user_neigh_conv_embedding.size(0)
        user_neigh_conv_embedding = user_neigh_conv_embedding.view(batch_size, -1)
        item_neigh_embedding = item_neigh_embedding.permute(0, 2, 1)
        item_neigh_conv_embedding = self.item_conv(item_neigh_embedding)
        # batch_size * out_channel * pool_size
        item_neigh_conv_embedding = item_neigh_conv_embedding.view(batch_size, -1)
        mf_vec = torch.mul(user_embedding, item_embedding)
        last = torch.cat((mf_vec, user_neigh_conv_embedding, item_neigh_conv_embedding), dim=-1)

        output = self.mlp_layers(last)
        out = self.out_layer(output)
        out = out.squeeze(-1)
        return out

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)