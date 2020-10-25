# -*- coding: utf-8 -*-
# @Time   : 2020/8/28 14:32
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/10/2
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

r"""
FPMC
################################################

Reference:
    Steffen Rendle et al. "Factorizing Personalized Markov Chains for Next-Basket Recommendation." in WWW 2010.

"""
import torch
from torch import nn
from torch.nn.init import xavier_normal_

from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender


class FPMC(SequentialRecommender):
    r"""The FPMC model is mainly used in the recommendation system to predict the possibility of
    unknown items arousing user interest, and to discharge the item recommendation list.

    Note:

        In order that the generation method we used is common to other sequential models,
        We set the size of the basket mentioned in the paper equal to 1.
        For comparison with other models, the loss function used is BPR.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FPMC, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']

        # load dataset info
        self.n_users = dataset.user_num

        # define layers and loss
        # user embedding matrix
        self.UI_emb = nn.Embedding(self.n_users, self.embedding_size)
        # label embedding matrix
        self.IU_emb = nn.Embedding(self.n_items, self.embedding_size)
        # last click item embedding matrix
        self.LI_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # label embedding matrix
        self.IL_emb = nn.Embedding(self.n_items, self.embedding_size)
        self.loss_fct = BPRLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, user, item_seq, item_seq_len, next_item):

        item_last_click_index = item_seq_len - 1
        item_last_click = torch.gather(item_seq, dim=1, index=item_last_click_index.unsqueeze(1))
        item_seq_emb = self.LI_emb(item_last_click) # [b,1,emb]

        user_emb = self.UI_emb(user)
        user_emb = torch.unsqueeze(user_emb, dim=1) # [b,1,emb]

        iu_emb = self.IU_emb(next_item)
        iu_emb = torch.unsqueeze(iu_emb, dim=1) # [b,n,emb] in here n = 1

        il_emb = self.IL_emb(next_item)
        il_emb = torch.unsqueeze(il_emb, dim=1) # [b,n,emb] in here n = 1

        # This is the core part of the FPMC model,can be expressed by a combination of a MF and a FMC model
        #  MF
        mf = torch.matmul(user_emb, iu_emb.permute(0, 2, 1))
        mf = torch.squeeze(mf, dim=1)  # [B,1]
        #  FMC
        fmc = torch.matmul(il_emb, item_seq_emb.permute(0, 2, 1))
        fmc = torch.squeeze(fmc, dim=1)  # [B,1]

        score = mf + fmc
        score = torch.squeeze(score)
        return score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        pos_score = self.forward(user, item_seq, item_seq_len, pos_items)
        neg_score = self.forward(user, item_seq, item_seq_len, neg_items)
        loss = self.loss_fct(pos_score, neg_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        score = self.forward(user, item_seq, item_seq_len, test_item)  # [B]
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        user_emb = self.UI_emb(user)
        all_iu_emb = self.IU_emb.weight
        mf = torch.matmul(user_emb, all_iu_emb.transpose(0,1))
        all_il_emb = self.IL_emb.weight

        item_last_click_index = item_seq_len - 1
        item_last_click = torch.gather(item_seq, dim=1, index=item_last_click_index.unsqueeze(1))
        item_seq_emb = self.LI_emb(item_last_click)  # [b,1,emb]
        fmc = torch.matmul(item_seq_emb, all_il_emb.transpose(0,1))
        fmc = torch.squeeze(fmc, dim=1)
        score = mf + fmc
        return score
