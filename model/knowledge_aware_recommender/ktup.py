# @Time   : 2020/8/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
Reference:
Yixin Cao et al., "Unifying Knowledge Graph Learning and Recommendation:Towards a Better Understanding of User Preferences." in WWW 2019.

Code Reference:
https://github.com/TaoMiner/joint-kg-recommender
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.autograd import Variable

from model.abstract_recommender import KnowledgeRecommender
from model.loss import BPRLoss, MarginLoss


# todo: L2 regularization
class KTUP(KnowledgeRecommender):

    def __init__(self, config, dataset):
        super(KTUP, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.use_st_gumbel = config['use_st_gumbel']
        self.margin = config['margin']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.pref_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.pref_norm_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.relation_norm_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        self.rec_loss = BPRLoss()
        self.kg_loss = MarginLoss(margin=self.margin)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def st_gumbel_softmax(self, logits, temperature=1.0):
        """
        Return the result of Straight-Through Gumbel-Softmax Estimation.
        It approximates the discrete sampling via Gumbel-Softmax trick
        and applies the biased ST estimator.
        In the forward propagation, it emits the discrete one-hot result,
        and in the backward propagation it approximates the categorical
        distribution via smooth Gumbel-Softmax distribution.
        Args:
            logits (Variable): A un-normalized probability values,
                which has the size (batch_size, num_classes)
            temperature (float): A temperature parameter. The higher
                the value is, the smoother the distribution is.
        Returns:
            y: The sampled output, which has the property explained above.
        """

        eps = 1e-20
        u = logits.data.new(*logits.size()).uniform_()
        gumbel_noise = Variable(-torch.log(-torch.log(u + eps) + eps))
        y = logits + gumbel_noise
        y = self.masked_softmax(logits=y / temperature)
        y_argmax = y.max(len(y.shape) - 1)[1]
        y_hard = self.convert_to_one_hot(
            indices=y_argmax,
            num_classes=y.size(len(y.shape) - 1)).float()
        y = (y_hard - y).detach() + y
        return y

    def get_prefernces(self, user_e, item_e, use_st_gumbel=False):
        pref_probs = torch.matmul(user_e + item_e, torch.t(self.pref_embedding.weight + self.relation_embedding.weight)) / 2
        if use_st_gumbel:
            # todo: different torch versions may cause the st_gumbel_softmax to report errors, wait to be test
            pref_probs = self.st_gumbel_softmax(pref_probs)
        relation_e = torch.matmul(pref_probs, self.pref_embedding.weight + self.relation_embedding) / 2
        norm_e = torch.matmul(pref_probs, self.pref_norm_embedding + self.relation_norm_embedding) / 2
        return pref_probs, relation_e, norm_e

    @staticmethod
    def transH_projection(original, norm):
        return original - torch.sum(original * norm, dim=len(original.size()) - 1, keepdim=True) * norm

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        entity_e = self.item_embedding(item)
        item_e = item_e + entity_e

        _, relation_e, norm_e = self.get_prefernces(user_e, item_e, use_st_gumbel=self.use_st_gumbel)
        proj_user_e = self.transH_projection(user_e, norm_e)
        proj_item_e = self.transH_projection(item_e, norm_e)

        return proj_user_e, relation_e, proj_item_e

    def calculate_loss(self, interaction, is_rec=True):
        if is_rec:
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]
            proj_user_e, pos_relation_e, proj_pos_item_e = self.forward(user, pos_item)
            proj_user_e, neg_relation_e, proj_neg_item_e = self.forwar(user, neg_item)

            pos_item_score = ((proj_user_e + pos_relation_e - proj_pos_item_e) ** 2).sum(dim=1)
            neg_item_score = ((proj_user_e + neg_relation_e - proj_neg_item_e) ** 2).sum(dim=1)

            rec_loss = - self.rec_loss(pos_item_score, neg_item_score)

            return rec_loss

        else:
            h = interaction[self.HEAD_ENTITY_ID]
            r = interaction[self.RELATION_ID]
            pos_t = interaction[self.TAIL_ENTITY_ID]
            neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

            h_e = self.entity_embedding(h)
            pos_t_e = self.entity_embedding(pos_t)
            neg_t_e = self.entity_embedding(neg_t)
            r_e = self.relation_embedding(r)
            norm_e = self.relation_norm_embedding(r)

            proj_h_e = self.transH_projection(h_e, norm_e)
            proj_pos_t_e = self.transH_projection(pos_t_e, norm_e)
            proj_neg_t_e = self.transH_projection(neg_t_e, norm_e)

            pos_tail_score = ((proj_h_e + r_e - proj_pos_t_e) ** 2).sum(dim=1)
            neg_tail_score = ((proj_h_e + r_e - proj_neg_t_e) ** 2).sum(dim=1)

            kg_loss = self.kg_loss(pos_tail_score, neg_tail_score)

            return kg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        proj_user_e, relation_e, proj_item_e = self.forward(user, item)
        return ((proj_user_e + relation_e - proj_item_e) ** 2).sum(dim=1)
