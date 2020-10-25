# -*- coding: utf-8 -*-
# @Time   : 2020/9/28
# @Author : gaole he
# @Email  : hegaole@ruc.edu.cn


r"""
RippleNet
#####################################################
Reference:
    Hongwei Wang et al. "RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems."
    in CIKM 2018.
"""

import torch
import torch.nn as nn
import numpy as np
import collections

from recbole.utils import InputType
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_normal_initialization


class RippleNet(KnowledgeRecommender):
    r"""RippleNet is an knowledge enhanced matrix factorization model.
    The original interaction matrix of :math:`n_{users} \times n_{items}`
    and related knowledge graph is set as model input,
    we carefully design the data interface and use ripple set to train and test efficiently.
    We just implement the model following the original author with a pointwise training mode.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RippleNet, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_weight = config['kg_weight']
        self.reg_weight = config['reg_weight']
        self.n_hop = config['n_hop']
        self.n_memory = config['n_memory']
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        head_entities = dataset.dataset.head_entities.tolist()
        tail_entities = dataset.dataset.tail_entities.tolist()
        relations = dataset.dataset.relations.tolist()
        kg = {}
        for i in range(len(head_entities)):
            head_ent = head_entities[i]
            tail_ent = tail_entities[i]
            relation = relations[i]
            kg.setdefault(head_ent, [])
            kg[head_ent].append((tail_ent, relation))
        self.kg = kg
        users = self.interaction_matrix.row.tolist()
        items = self.interaction_matrix.col.tolist()
        user_dict = {}
        for i in range(len(users)):
            user = users[i]
            item = items[i]
            user_dict.setdefault(user, [])
            user_dict[user].append(item)
        self.user_dict = user_dict
        self.ripple_set = self._build_ripple_set()

        # define layers and loss
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size * self.embedding_size)
        self.transform_matrix = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.rec_loss = BPRLoss()
        self.l2_loss = EmbLoss()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _build_ripple_set(self):
        r"""Get the normalized interaction matrix of users and items according to A_values.
        Get the ripple hop-wise ripple set for every user, w.r.t. their interaction history

        Returns:
            ripple_set (dict)
        """
        ripple_set = collections.defaultdict(list)
        n_padding = 0
        for user in self.user_dict:
            for h in range(self.n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = self.user_dict[user]
                else:
                    tails_of_last_hop = ripple_set[user][-1][2]

                for entity in tails_of_last_hop:
                    if entity not in self.kg:
                        continue
                    for tail_and_relation in self.kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                # if the current ripple set of the given user is empty,
                # we simply copy the ripple set of the last hop here
                if len(memories_h) == 0:
                    if h == 0:
                        # print("user {} without 1-hop kg facts, fill with padding".format(user))
                        # raise AssertionError("User without facts in 1st hop")
                        n_padding += 1
                        memories_h = [0 for i in range(self.n_memory)]
                        memories_r = [0 for i in range(self.n_memory)]
                        memories_t = [0 for i in range(self.n_memory)]
                        memories_h = torch.LongTensor(memories_h).to(self.device)
                        memories_r = torch.LongTensor(memories_r).to(self.device)
                        memories_t = torch.LongTensor(memories_t).to(self.device)
                        ripple_set[user].append((memories_h, memories_r, memories_t))
                    else:
                        ripple_set[user].append(ripple_set[user][-1])
                else:
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < self.n_memory
                    indices = np.random.choice(len(memories_h), size=self.n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    memories_h = torch.LongTensor(memories_h).to(self.device)
                    memories_r = torch.LongTensor(memories_r).to(self.device)
                    memories_t = torch.LongTensor(memories_t).to(self.device)
                    ripple_set[user].append((memories_h, memories_r, memories_t))
        print("{} among {} users are padded".format(n_padding, len(self.user_dict)))
        return ripple_set

    def forward(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()
        memories_h, memories_r, memories_t = {}, {}, {}
        for hop in range(self.n_hop):
            memories_h[hop] = []
            memories_r[hop] = []
            memories_t[hop] = []
            for user in users:
                memories_h[hop].append(self.ripple_set[user][hop][0])
                memories_r[hop].append(self.ripple_set[user][hop][1])
                memories_t[hop].append(self.ripple_set[user][hop][2])
        # memories_h, memories_r, memories_t = self.ripple_set[user]
        item = interaction[self.ITEM_ID]
        self.item_embeddings = self.entity_embedding(item)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size * n_memory]
            head_ent = torch.cat(memories_h[i], dim=0)
            relation = torch.cat(memories_r[i], dim=0)
            tail_ent = torch.cat(memories_t[i], dim=0)
            # print("Hop {}, size {}".format(i, head_ent.size(), relation.size(), tail_ent.size()))

            # [batch size * n_memory, dim]
            self.h_emb_list.append(self.entity_embedding(head_ent))

            # [batch size * n_memory, dim * dim]
            self.r_emb_list.append(self.relation_embedding(relation))

            # [batch size * n_memory, dim]
            self.t_emb_list.append(self.entity_embedding(tail_ent))

        o_list = self._key_addressing()
        y = o_list[-1]
        for i in range(self.n_hop - 1):
            y = y + o_list[i]
        scores = torch.sum(self.item_embeddings * y, dim=1)
        return scores

    def _key_addressing(self):
        r"""Conduct reasoning for specific item and user ripple set

        Returns:
            o_list (dict -> torch.cuda.FloatTensor): list of torch.cuda.FloatTensor n_hop * [batch_size, embedding_size]
        """
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size * n_memory, dim, 1]
            h_emb = self.h_emb_list[hop].unsqueeze(2)

            # [batch_size * n_memory, dim, dim]
            r_mat = self.r_emb_list[hop].view(-1, self.embedding_size, self.embedding_size)
            # [batch_size, n_memory, dim]
            Rh = torch.bmm(r_mat, h_emb).view(-1, self.n_memory, self.embedding_size)

            # [batch_size, dim, 1]
            v = self.item_embeddings.unsqueeze(2)

            # [batch_size, n_memory]
            probs = torch.bmm(Rh, v).squeeze(2)

            # [batch_size, n_memory]
            probs_normalized = self.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = probs_normalized.unsqueeze(2)

            tail_emb = self.t_emb_list[hop].view(-1, self.n_memory, self.embedding_size)

            # [batch_size, dim]
            o = torch.sum(tail_emb * probs_expanded, dim=1)

            self.item_embeddings = self.transform_matrix(self.item_embeddings + o)
            # item embedding update
            o_list.append(o)
        return o_list

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        rec_loss = self.loss(output, label)

        kge_loss = None
        for hop in range(self.n_hop):
            # (batch_size * n_memory, 1, dim)
            h_expanded = self.h_emb_list[hop].unsqueeze(1)
            # (batch_size * n_memory, dim)
            t_expanded = self.t_emb_list[hop]
            # (batch_size * n_memory, dim, dim)
            r_mat = self.r_emb_list[hop].view(-1, self.embedding_size, self.embedding_size)
            # (N, 1, dim) (N, dim, dim) -> (N, 1, dim)
            hR = torch.bmm(h_expanded, r_mat).squeeze(1)
            # (N, dim) (N, dim)
            hRt = torch.sum(hR * t_expanded, dim=1)
            if kge_loss is None:
                kge_loss = torch.mean(self.sigmoid(hRt))
            else:
                kge_loss = kge_loss + torch.mean(self.sigmoid(hRt))

        reg_loss = None
        for hop in range(self.n_hop):
            tp_loss = self.l2_loss(self.h_emb_list[hop], self.t_emb_list[hop], self.r_emb_list[hop])
            if reg_loss is None:
                reg_loss = tp_loss
            else:
                reg_loss = reg_loss + tp_loss
        reg_loss = reg_loss + self.l2_loss(self.transform_matrix.weight)
        loss = rec_loss - self.kg_weight * kge_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        scores = self.forward(interaction)
        return scores

    def _key_addressing_full(self):
        r"""Conduct reasoning for specific item and user ripple set

        Returns:
            o_list (dict -> torch.cuda.FloatTensor): list of torch.cuda.FloatTensor n_hop * [batch_size, n_item, embedding_size]
        """
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size * n_memory, dim, 1]
            h_emb = self.h_emb_list[hop].unsqueeze(2)

            # [batch_size * n_memory, dim, dim]
            r_mat = self.r_emb_list[hop].view(-1, self.embedding_size, self.embedding_size)
            # [batch_size, n_memory, dim]
            Rh = torch.bmm(r_mat, h_emb).view(-1, self.n_memory, self.embedding_size)

            batch_size = Rh.size(0)

            if len(self.item_embeddings.size()) == 2:
                # [1, n_item, dim]
                self.item_embeddings = self.item_embeddings.unsqueeze(0)
                # [batch_size, n_item, dim]
                self.item_embeddings = self.item_embeddings.expand(batch_size, -1, -1)
                # [batch_size, dim, n_item]
                v = self.item_embeddings.transpose(1, 2)
                # [batch_size, dim, n_item]
                v = v.expand(batch_size, -1, -1)
            else:
                assert len(self.item_embeddings.size()) == 3
                # [batch_size, dim, n_item]
                v = self.item_embeddings.transpose(1, 2)

            # [batch_size, n_memory, n_item]
            probs = torch.bmm(Rh, v)

            # [batch_size, n_memory, n_item]
            probs_normalized = self.softmax(probs)

            # [batch_size, n_item, n_memory]
            probs_transposed = probs_normalized.transpose(1, 2)

            # [batch_size, n_memory, dim]
            tail_emb = self.t_emb_list[hop].view(-1, self.n_memory, self.embedding_size)

            # [batch_size, n_item, dim]
            o = torch.bmm(probs_transposed, tail_emb)

            # [batch_size, n_item, dim] [batch_size, n_item, dim] -> [batch_size, n_item, dim]
            self.item_embeddings = self.transform_matrix(self.item_embeddings + o)
            # item embedding update
            o_list.append(o)
        return o_list

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()
        memories_h, memories_r, memories_t = {}, {}, {}
        for hop in range(self.n_hop):
            memories_h[hop] = []
            memories_r[hop] = []
            memories_t[hop] = []
            for user in users:
                memories_h[hop].append(self.ripple_set[user][hop][0])
                memories_r[hop].append(self.ripple_set[user][hop][1])
                memories_t[hop].append(self.ripple_set[user][hop][2])
        # memories_h, memories_r, memories_t = self.ripple_set[user]
        # item = interaction[self.ITEM_ID]
        self.item_embeddings = self.entity_embedding.weight[:self.n_items]
        # self.item_embeddings = self.entity_embedding(item)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size * n_memory]
            head_ent = torch.cat(memories_h[i], dim=0)
            relation = torch.cat(memories_r[i], dim=0)
            tail_ent = torch.cat(memories_t[i], dim=0)
            # print("Hop {}, size {}".format(i, head_ent.size(), relation.size(), tail_ent.size()))

            # [batch size * n_memory, dim]
            self.h_emb_list.append(self.entity_embedding(head_ent))

            # [batch size * n_memory, dim * dim]
            self.r_emb_list.append(self.relation_embedding(relation))

            # [batch size * n_memory, dim]
            self.t_emb_list.append(self.entity_embedding(tail_ent))

        o_list = self._key_addressing_full()
        y = o_list[-1]
        for i in range(self.n_hop - 1):
            y = y + o_list[i]
        # [batch_size, n_item, dim] [batch_size, n_item, dim]
        scores = torch.sum(self.item_embeddings * y, dim=-1)
        return scores.view(-1)
