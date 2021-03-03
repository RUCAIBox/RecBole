# -*- coding: utf-8 -*-
# @Time   : 2021/1/14
# @Author : Chengyuan Li
# @Email  : 2017202049@ruc.edu.cn

r"""
NNCF
################################################
Reference:
    Ting Bai et al. "A Neural Collaborative Filtering Model with Interaction-based Neighborhood." in CIKM 2017.

Reference code:
    https://github.com/Tbbaby/NNCF-Pytorch

"""

import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType

import numpy as np
from sklearn.metrics import jaccard_score

class ComputeSimilarity:

    def __init__(self, dataMatrix, topk):
        r"""Compute the similarity of users and items.
        Args:
            dataMatrix (scipy.sparse.coo_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
        """

        super(ComputeSimilarity, self).__init__()

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topk, self.n_columns)
        self.dataMatrix = dataMatrix.copy().tocsr()

    def compute_similarity(self, block_size=100):
        r"""Compute the similarity for the given dataset.

        Args:
            block_size(int): divide matrix to :math:`n\_columns \div block\_size` to calculate cosine_distance

        Returns:
            list: The similar nodes of users, shape: [number of users, neigh_num]
            list: The similar nodes of items, shape: [number of items, neigh_num]
        """
        u_neigh = []
        i_neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        sumOfSquared_u = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
        sumOfSquared_u = np.sqrt(sumOfSquared_u)
        sumOfSquared_i = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
        sumOfSquared_i = np.sqrt(sumOfSquared_i)

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

                # Apply normalization, ensure denominator != 0
                denominator = sumOfSquared_i[columnIndex] * sumOfSquared_i + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]
                i_neigh.append(top_k_idx)

            start_col_block += block_size
        # End while on columns

        end_row_local = self.n_rows
        start_row_block = 0

        # Compute all similarities for each user using vectorization
        while start_row_block < end_row_local:

            end_row_block = min(start_row_block + block_size, end_row_local)
            this_block_size = end_row_block - start_row_block

            # All data points for a given user
            user_data = self.dataMatrix[start_row_block:end_row_block, :]
            user_data = user_data.toarray().squeeze()

            if user_data.ndim == 1:
                user_data = np.expand_dims(user_data, axis=1)

            # Compute user similarities
            this_block_weights = self.dataMatrix.dot(user_data.T)
            for row_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_row_weights = this_row_weights.squeeze()
                else:
                    this_row_weights = this_block_weights[:, row_index_in_block]

                rowIndex = row_index_in_block + start_row_block
                this_row_weights[rowIndex] = 0.0

                # Apply normalization, ensure denominator != 0
                denominator = sumOfSquared_u[rowIndex] * sumOfSquared_u + 1e-6
                this_row_weights = np.multiply(this_row_weights, 1 / denominator)

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of users
                # - Partition the data to extract the set of relevant users
                # - Sort only the relevant users
                # - Get the original user index
                relevant_users_partition = (-this_row_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_users_partition_sorting = np.argsort(-this_row_weights[relevant_users_partition])
                top_k_idx = relevant_users_partition[relevant_users_partition_sorting]
                u_neigh.append(top_k_idx)

            start_row_block += block_size
            # End while on rows

        return u_neigh, i_neigh

class NNCF(GeneralRecommender):
    r"""NNCF is an neural network enhanced matrix factorization model which also captures neighborhood information.
    We implement the NNCF model with three ways to process neighborhood information.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NNCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.ui_embedding_size = config['ui_embedding_size']
        self.neigh_embedding_size = config['neigh_embedding_size']
        self.num_conv_kernel = config['num_conv_kernel']
        self.conv_kernel_size = config['conv_kernel_size']
        self.pool_kernel_size = config['pool_kernel_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.neigh_num = config['neigh_num']
        self.neigh_info_method = config['neigh_info_method']
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
        self.out_layer = nn.Sequential(nn.Linear(self.mlp_hidden_size[-1], 1),
                                       nn.Sigmoid())
        self.dropout_layer = torch.nn.Dropout(p=config['dropout'])
        self.loss = nn.BCELoss()

        # choose the method to use neighborhood information
        if self.neigh_info_method == "random":
            self.u_neigh, self.i_neigh = self.get_neigh_random()
        elif self.neigh_info_method == "knn":
            self.u_neigh, self.i_neigh = self.get_neigh_knn()
        elif self.neigh_info_method == "louvain":
            self.u_neigh, self.i_neigh = self.get_neigh_louvain()
        else:
            raise RuntimeError('You need to choose the right algorithm of processing neighborhood information. \
                The parameter neigh_info_method can be set to random, knn or louvain.')

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    # Unify embedding length
    def Max_ner(self, lst, max_ner):
        r"""Unify embedding length of neighborhood information for efficiency consideration. 
        Truncate the list if the length is larger than max_ner.
        Otherwise, pad it with 0. 

        Args:
            lst (list): The input list contains node's neighbors.
            max_ner (int): The number of neighbors we choose for each node.

        Returns:
            list: The list of a node's community neighbors.


        """
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
        r"""Find other nodes in the same community. 
        e.g. If the node starts with letter "i", 
        the other nodes start with letter "i" in the same community dict group are its community neighbors.

        Args:
            partition (dict): The input dict that contains the community each node belongs.
            community_dict (dict): The input dict that shows the nodes each community contains.
            node (int): The id of the input node.
            kind (char): The type of the input node.

        Returns:
            list: The list of a node's community neighbors.

        """
        comm = community_dict[partition[node]]
        return [x for x in comm if x.startswith(kind)]

    # Prepare neiborhood embeddings, i.e. I(u) and U(i)
    def prepare_vector_element(self, partition, relation, community_dict):
        r"""Find the community neighbors of each node, i.e. I(u) and U(i).
        Then reset the id of nodes.

        Args:
            partition (dict): The input dict that contains the community each node belongs.
            relation (list): The input list that contains the relationships of users and items.
            community_dict (dict): The input dict that shows the nodes each community contains.

        Returns:
            list: The list of nodes' community neighbors.

        """
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
        r"""Get neighborhood information using louvain algorithm.
        First, change the id of node, 
        for example, the id of user node "1" will be set to "u_1" in order to use louvain algorithm.
        Second, use louvain algorithm to seperate nodes into different communities.
        Finally, find the community neighbors of each node with the same type and reset the id of the nodes.

        Returns:
            torch.IntTensor: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
        """
        inter_M = self.interaction_matrix
        pairs = list(zip(inter_M.row, inter_M.col))

        tmp_relation = []
        for i in range(len(pairs)):
            tmp_relation.append(['user_' + str(pairs[i][0]), 'item_' + str(pairs[i][1])])
        
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(tmp_relation)
        resolution = self.resolution
        import community
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

    # Get neighborhood embeddings using knn method
    def get_neigh_knn(self):
        r"""Get neighborhood information using knn algorithm.
        Find direct neighbors of each node, if the number of direct neighbors is less than neigh_num, 
        add other similar neighbors using knn algorithm.
        Otherwise, select random top k direct neighbors, k equals to the number of neighbors. 

        Returns:
            torch.IntTensor: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
        """
        inter_M = self.interaction_matrix
        pairs = list(zip(inter_M.row, inter_M.col))
        ui_inters = np.zeros((self.n_users, self.n_items), dtype=np.int8)

        for i in range(len(pairs)):
            ui_inters[pairs[i][0], pairs[i][1]] = 1

        # Get similar neighbors using knn algorithm
        user_knn, item_knn = ComputeSimilarity(self.interaction_matrix, topk=self.neigh_num).compute_similarity()

        u_neigh, i_neigh = [], []

        for u in range(self.n_users):
            neigh_list = ui_inters[u].nonzero()[0]
            direct_neigh_num = len(neigh_list)
            if len(neigh_list) == 0:
                u_neigh.append(self.neigh_num * [0])
            elif direct_neigh_num < self.neigh_num:
                tmp_k = self.neigh_num - direct_neigh_num
                mask = np.random.randint(0, len(neigh_list), size=1)
                neigh_list = list(neigh_list) + list(item_knn[neigh_list[mask[0]]])
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
                mask = np.random.randint(0, len(neigh_list), size=1)
                neigh_list = list(neigh_list) + list(user_knn[neigh_list[mask[0]]])
                i_neigh.append(neigh_list[:self.neigh_num])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                i_neigh.append(neigh_list[mask])

        u_neigh = torch.tensor(u_neigh, device=self.device)
        i_neigh = torch.tensor(i_neigh, device=self.device)
        return u_neigh, i_neigh

    # Get neighborhood embeddings using random method
    def get_neigh_random(self):
        r"""Get neighborhood information using random algorithm.
        Select random top k direct neighbors, k equals to the number of neighbors. 
        
        Returns:
            torch.IntTensor: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
        """
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
        r"""Get a batch of neighborhood embedding tensor according to input id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The neighborhood embedding tensor of a batch of user, shape: [batch_size, neigh_embedding_size]
            torch.FloatTensor: The neighborhood embedding tensor of a batch of item, shape: [batch_size, neigh_embedding_size]

        """
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
