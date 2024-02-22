# -*- coding: utf-8 -*-
# @Time   : 2020/10/3
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

r"""
KGNNLS
################################################

Reference:
    Hongwei Wang et al. "Knowledge-aware Graph Neural Networks with Label Smoothness Regularization
    for Recommender Systems." in KDD 2019.

Reference code:
    https://github.com/hwwang55/KGNN-LS
"""

import random

import numpy as np
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class KGNNLS(KnowledgeRecommender):
    r"""KGNN-LS is a knowledge-based recommendation model.
    KGNN-LS transforms the knowledge graph into a user-specific weighted graph and then apply a graph neural network to
    compute personalized item embeddings. To provide better inductive bias, KGNN-LS relies on label smoothness
    assumption, which posits that adjacent items in the knowledge graph are likely to have similar user relevance
    labels/scores. Label smoothness provides regularization over the edge weights and it is equivalent  to a label
    propagation scheme on a graph.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGNNLS, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.neighbor_sample_size = config["neighbor_sample_size"]
        self.aggregator_class = config["aggregator"]  # which aggregator to use
        # number of iterations when computing entity representation
        self.n_iter = config["n_iter"]
        self.reg_weight = config["reg_weight"]  # weight of l2 regularization
        # weight of label Smoothness regularization
        self.ls_weight = config["ls_weight"]

        # define embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(
            self.n_relations + 1, self.embedding_size
        )

        # sample neighbors and construct interaction table
        kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")
        adj_entity, adj_relation = self.construct_adj(kg_graph)
        self.adj_entity, self.adj_relation = adj_entity.to(
            self.device
        ), adj_relation.to(self.device)

        inter_feat = dataset.inter_feat
        pos_users = inter_feat[dataset.uid_field]
        pos_items = inter_feat[dataset.iid_field]
        pos_label = torch.ones(pos_items.shape)
        pos_interaction_table, self.offset = self.get_interaction_table(
            pos_users, pos_items, pos_label
        )
        self.interaction_table = self.sample_neg_interaction(
            pos_interaction_table, self.offset
        )

        # define function
        self.softmax = nn.Softmax(dim=-1)
        self.linear_layers = torch.nn.ModuleList()
        for i in range(self.n_iter):
            self.linear_layers.append(
                nn.Linear(
                    (
                        self.embedding_size
                        if not self.aggregator_class == "concat"
                        else self.embedding_size * 2
                    ),
                    self.embedding_size,
                )
            )
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l2_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["adj_entity", "adj_relation"]

    def get_interaction_table(self, user_id, item_id, y):
        r"""Get interaction_table that is used for fetching user-item interaction label in LS regularization.

        Args:
            user_id(torch.Tensor): the user id in user-item interactions, shape: [n_interactions, 1]
            item_id(torch.Tensor): the item id in user-item interactions, shape: [n_interactions, 1]
            y(torch.Tensor): the label in user-item interactions, shape: [n_interactions, 1]

        Returns:
            tuple:
                - interaction_table(dict): key: user_id * 10^offset + item_id; value: y_{user_id, item_id}
                - offset(int): The offset that is used for calculating the key(index) in interaction_table
        """
        offset = len(str(self.n_entities))
        offset = 10**offset
        keys = user_id * offset + item_id
        keys = keys.int().cpu().numpy().tolist()
        values = y.float().cpu().numpy().tolist()

        interaction_table = dict(zip(keys, values))
        return interaction_table, offset

    def sample_neg_interaction(self, pos_interaction_table, offset):
        r"""Sample neg_interaction to construct train data.

        Args:
            pos_interaction_table(dict): the interaction_table that only contains pos_interaction.
            offset(int): The offset that is used for calculating the key(index) in interaction_table

        Returns:
            interaction_table(dict): key: user_id * 10^offset + item_id; value: y_{user_id, item_id}
        """
        pos_num = len(pos_interaction_table)
        neg_num = 0
        neg_interaction_table = {}
        while neg_num < pos_num:
            user_id = random.randint(0, self.n_users)
            item_id = random.randint(0, self.n_items)
            keys = user_id * offset + item_id
            if keys not in pos_interaction_table:
                neg_interaction_table[keys] = 0.0
                neg_num += 1
        interaction_table = {**pos_interaction_table, **neg_interaction_table}
        return interaction_table

    def construct_adj(self, kg_graph):
        r"""Get neighbors and corresponding relations for each entity in the KG.

        Args:
            kg_graph(scipy.sparse.coo_matrix): an undirected graph

        Returns:
            tuple:
                - adj_entity (torch.LongTensor): each line stores the sampled neighbor entities for a given entity,
                  shape: [n_entities, neighbor_sample_size]
                - adj_relation (torch.LongTensor): each line stores the corresponding sampled neighbor relations,
                  shape: [n_entities, neighbor_sample_size]
        """
        # self.logger.info('constructing knowledge graph ...')
        # treat the KG as an undirected graph
        kg_dict = dict()
        for triple in zip(kg_graph.row, kg_graph.data, kg_graph.col):
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            if head not in kg_dict:
                kg_dict[head] = []
            kg_dict[head].append((tail, relation))
            if tail not in kg_dict:
                kg_dict[tail] = []
            kg_dict[tail].append((head, relation))

        # self.logger.info('constructing adjacency matrix ...')
        # each line of adj_entity stores the sampled neighbor entities for a given entity
        # each line of adj_relation stores the corresponding sampled neighbor relations
        entity_num = kg_graph.shape[0]
        adj_entity = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        for entity in range(entity_num):
            if entity not in kg_dict.keys():
                adj_entity[entity] = np.array([entity] * self.neighbor_sample_size)
                adj_relation[entity] = np.array([0] * self.neighbor_sample_size)
                continue

            neighbors = kg_dict[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= self.neighbor_sample_size:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=False,
                )
            else:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=True,
                )
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

        return torch.from_numpy(adj_entity), torch.from_numpy(adj_relation)

    def get_neighbors(self, items):
        r"""Get neighbors and corresponding relations for each entity in items from adj_entity and adj_relation.

        Args:
            items(torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            tuple:
                - entities(list): Entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                  dimensions of entities: {[batch_size, 1],
                  [batch_size, n_neighbor],
                  [batch_size, n_neighbor^2],
                  ...,
                  [batch_size, n_neighbor^n_iter]}
                - relations(list): Relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for
                  entities. Relations have the same shape as entities.
        """
        items = torch.unsqueeze(items, dim=1)
        entities = [items]
        relations = []
        for i in range(self.n_iter):
            index = torch.flatten(entities[i])
            neighbor_entities = torch.index_select(self.adj_entity, 0, index).reshape(
                self.batch_size, -1
            )
            neighbor_relations = torch.index_select(
                self.adj_relation, 0, index
            ).reshape(self.batch_size, -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, user_embeddings, entities, relations):
        r"""For each item, aggregate the entity representation and its neighborhood representation into a single vector.

        Args:
            user_embeddings(torch.FloatTensor): The embeddings of users, shape: [batch_size, embedding_size]
            entities(list): entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                            dimensions of entities: {[batch_size, 1],
                            [batch_size, n_neighbor],
                            [batch_size, n_neighbor^2],
                            ...,
                            [batch_size, n_neighbor^n_iter]}
            relations(list): relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                             relations have the same shape as entities.

        Returns:
            item_embeddings(torch.FloatTensor): The embeddings of items, shape: [batch_size, embedding_size]

        """
        entity_vectors = [self.entity_embedding(i) for i in entities]
        relation_vectors = [self.relation_embedding(i) for i in relations]

        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = (
                    self.batch_size,
                    -1,
                    self.neighbor_sample_size,
                    self.embedding_size,
                )
                self_vectors = entity_vectors[hop]
                neighbor_vectors = entity_vectors[hop + 1].reshape(shape)
                neighbor_relations = relation_vectors[hop].reshape(shape)

                # mix_neighbor_vectors
                user_embeddings = user_embeddings.reshape(
                    self.batch_size, 1, 1, self.embedding_size
                )  # [batch_size, 1, 1, dim]
                user_relation_scores = torch.mean(
                    user_embeddings * neighbor_relations, dim=-1
                )  # [batch_size, -1, n_neighbor]
                user_relation_scores_normalized = torch.unsqueeze(
                    self.softmax(user_relation_scores), dim=-1
                )  # [batch_size, -1, n_neighbor, 1]
                neighbors_agg = torch.mean(
                    user_relation_scores_normalized * neighbor_vectors, dim=2
                )  # [batch_size, -1, dim]

                if self.aggregator_class == "sum":
                    output = (self_vectors + neighbors_agg).reshape(
                        -1, self.embedding_size
                    )  # [-1, dim]
                elif self.aggregator_class == "neighbor":
                    output = neighbors_agg.reshape(-1, self.embedding_size)  # [-1, dim]
                elif self.aggregator_class == "concat":
                    # [batch_size, -1, dim * 2]
                    output = torch.cat([self_vectors, neighbors_agg], dim=-1)
                    output = output.reshape(
                        -1, self.embedding_size * 2
                    )  # [-1, dim * 2]
                else:
                    raise Exception("Unknown aggregator: " + self.aggregator_class)

                output = self.linear_layers[i](output)
                # [batch_size, -1, dim]
                output = output.reshape(self.batch_size, -1, self.embedding_size)

                if i == self.n_iter - 1:
                    vector = self.Tanh(output)
                else:
                    vector = self.ReLU(output)

                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = entity_vectors[0].reshape(self.batch_size, self.embedding_size)
        return res

    def label_smoothness_predict(self, user_embeddings, user, entities, relations):
        r"""Predict the label of items by label smoothness.

        Args:
            user_embeddings(torch.FloatTensor): The embeddings of users, shape: [batch_size*2, embedding_size],
            user(torch.FloatTensor): the index of users, shape: [batch_size*2]
            entities(list): entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                            dimensions of entities: {[batch_size*2, 1],
                            [batch_size*2, n_neighbor],
                            [batch_size*2, n_neighbor^2],
                            ...,
                            [batch_size*2, n_neighbor^n_iter]}
            relations(list): relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                             relations have the same shape as entities.

        Returns:
            predicted_labels(torch.FloatTensor): The predicted label of items, shape: [batch_size*2]
        """
        # calculate initial labels; calculate updating masks for label propagation
        entity_labels = []
        # True means the label of this item is reset to initial value during label propagation
        reset_masks = []
        holdout_item_for_user = None

        for entities_per_iter in entities:
            users = torch.unsqueeze(user, dim=1)  # [batch_size, 1]
            user_entity_concat = (
                users * self.offset + entities_per_iter
            )  # [batch_size, n_neighbor^i]

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            def lookup_interaction_table(x, _):
                x = int(x)
                label = self.interaction_table.setdefault(x, 0.5)
                return label

            initial_label = user_entity_concat.clone().cpu().double()
            initial_label.map_(initial_label, lookup_interaction_table)
            initial_label = initial_label.float().to(self.device)

            # False if the item is held out
            holdout_mask = (holdout_item_for_user - user_entity_concat).bool()
            # True if the entity is a labeled item
            reset_mask = (initial_label - 0.5).bool()
            reset_mask = torch.logical_and(
                reset_mask, holdout_mask
            )  # remove held-out items
            initial_label = (
                holdout_mask.float() * initial_label
                + torch.logical_not(holdout_mask).float() * 0.5
            )  # label initialization

            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        # we do not need the reset_mask for the last iteration
        reset_masks = reset_masks[:-1]

        # label propagation
        relation_vectors = [self.relation_embedding(i) for i in relations]
        for i in range(self.n_iter):
            entity_labels_next_iter = []
            for hop in range(self.n_iter - i):
                masks = reset_masks[hop]
                self_labels = entity_labels[hop]
                neighbor_labels = entity_labels[hop + 1].reshape(
                    self.batch_size, -1, self.neighbor_sample_size
                )
                neighbor_relations = relation_vectors[hop].reshape(
                    self.batch_size, -1, self.neighbor_sample_size, self.embedding_size
                )

                # mix_neighbor_labels
                user_embeddings = user_embeddings.reshape(
                    self.batch_size, 1, 1, self.embedding_size
                )  # [batch_size, 1, 1, dim]
                user_relation_scores = torch.mean(
                    user_embeddings * neighbor_relations, dim=-1
                )  # [batch_size, -1, n_neighbor]
                user_relation_scores_normalized = self.softmax(
                    user_relation_scores
                )  # [batch_size, -1, n_neighbor]

                neighbors_aggregated_label = torch.mean(
                    user_relation_scores_normalized * neighbor_labels, dim=2
                )  # [batch_size, -1, dim] # [batch_size, -1]
                output = (
                    masks.float() * self_labels
                    + torch.logical_not(masks).float() * neighbors_aggregated_label
                )

                entity_labels_next_iter.append(output)
            entity_labels = entity_labels_next_iter

        predicted_labels = entity_labels[0].squeeze(-1)
        return predicted_labels

    def forward(self, user, item):
        self.batch_size = item.shape[0]
        # [batch_size, dim]
        user_e = self.user_embedding(user)
        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items. dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(item)
        # [batch_size, dim]
        item_e = self.aggregate(user_e, entities, relations)

        return user_e, item_e

    def calculate_ls_loss(self, user, item, target):
        r"""Calculate label smoothness loss.

        Args:
            user(torch.FloatTensor): the index of users, shape: [batch_size*2],
            item(torch.FloatTensor): the index of items, shape: [batch_size*2],
            target(torch.FloatTensor): the label of user-item, shape: [batch_size*2],

        Returns:
            ls_loss: label smoothness loss
        """
        user_e = self.user_embedding(user)
        entities, relations = self.get_neighbors(item)

        predicted_labels = self.label_smoothness_predict(
            user_e, user, entities, relations
        )
        ls_loss = self.bce_loss(predicted_labels, target)
        return ls_loss

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        target = torch.zeros(len(user) * 2, dtype=torch.float32).to(self.device)
        target[: len(user)] = 1

        users = torch.cat((user, user))
        items = torch.cat((pos_item, neg_item))

        user_e, item_e = self.forward(users, items)
        predict = torch.mul(user_e, item_e).sum(dim=1)
        rec_loss = self.bce_loss(predict, target)

        ls_loss = self.calculate_ls_loss(users, items, target)
        l2_loss = self.l2_loss(user_e, item_e)

        loss = rec_loss + self.ls_weight * ls_loss + self.reg_weight * l2_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user_index = interaction[self.USER_ID]
        item_index = torch.tensor(range(self.n_items)).to(self.device)

        user = torch.unsqueeze(user_index, dim=1).repeat(1, item_index.shape[0])
        user = torch.flatten(user)
        item = torch.unsqueeze(item_index, dim=0).repeat(user_index.shape[0], 1)
        item = torch.flatten(item)

        user_e, item_e = self.forward(user, item)
        score = torch.mul(user_e, item_e).sum(dim=1)

        return score.view(-1)
