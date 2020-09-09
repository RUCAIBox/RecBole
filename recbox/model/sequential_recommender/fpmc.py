# @Time   : 2020/8/28 14:32
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

import torch
from torch import nn
from torch.nn.init import xavier_normal_
from ...utils import InputType
from ..loss import BPRLoss
from ..abstract_recommender import SequentialRecommender


class FPMC(SequentialRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(FPMC, self).__init__()
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.TARGET_ITEM_ID = config['TARGET_PREFIX'] + self.ITEM_ID
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']

        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.user_count = dataset.user_num
        self.item_count = dataset.item_num

        self.embedding_size = config['embedding_size']

        self.UI_emb = nn.Embedding(self.user_count, self.embedding_size)                # user emb
        self.IU_emb = nn.Embedding(self.item_count, self.embedding_size)                # pred emb
        self.LI_emb = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0) # item list emb
        self.IL_emb = nn.Embedding(self.item_count, self.embedding_size)                # pred emb
        self.loss = BPRLoss()


        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)


    def forward(self, interaction):
        item_id_list = interaction[self.ITEM_ID_LIST]
        # index = interaction[self.ITEM_LIST_LEN]
        # index = index.view(item_id_list.size(0), 1)
        # # reset masked_id to 0
        # item_id_list.scatter_(dim=1, index=index, src=torch.zeros_like(item_id_list))
        item_list_emb = self.LI_emb(item_id_list) # [b,n,emb]

        user_emb = self.UI_emb(interaction[self.USER_ID])
        user_emb = torch.unsqueeze(user_emb, dim=1) # [b,1,emb]

        pos_iu = self.IU_emb(interaction[self.TARGET_ITEM_ID])
        pos_iu = torch.unsqueeze(pos_iu, dim=1) # [b,1,emb]

        pos_il = self.IL_emb(interaction[self.TARGET_ITEM_ID])
        pos_il = torch.unsqueeze(pos_il, dim=1) # [b,1,emb]

        pos_score = self.pmfc(user_emb, pos_iu, pos_il, item_list_emb)

        pos_score = torch.squeeze(pos_score)
        return pos_score



    def pmfc(self, Vui, Viu, Vil, Vli):
        """
        :param Vui: user embedding:[B,1,E]
        :param Viu: pos or neg:[B,1,E]
        :param Vil: pos or neg:[B,1,E]
        :param Vli: item_list :[B,S,E]
        :return:
        """
    #     MF
        mf = torch.matmul(Vui, Viu.permute(0, 2, 1))
        mf = torch.squeeze(mf, dim=1)#[B,1]

    #     PMC
        pmf = torch.matmul(Vil, Vli.permute(0, 2, 1))
        pmf = torch.mean(pmf, dim=-1)#[B,1]
        x = mf + pmf
        return x



    def calculate_loss(self, interaction):
        user_emb = self.UI_emb(interaction[self.USER_ID])
        user_emb = torch.unsqueeze(user_emb, dim=1)  # [b,1,emb]

        item_id_list = interaction[self.ITEM_ID_LIST]
        # index = interaction[self.ITEM_LIST_LEN]
        # index = index.view(item_id_list.size(0), 1)
        # reset masked_id to 0
        # item_id_list.scatter_(dim=1, index=index, src=torch.zeros_like(item_id_list))
        item_list_emb = self.LI_emb(item_id_list)  # [b,n,emb]

        neg_item = interaction[self.NEG_ITEM_ID]
        neg_iu = self.IU_emb(neg_item)
        neg_iu = torch.unsqueeze(neg_iu, dim=1)  # [b,1,emb]

        neg_il = self.IL_emb(neg_item)
        neg_il = torch.unsqueeze(neg_il, dim=1)  # [b,1,emb]

        neg_score = self.pmfc(user_emb, neg_iu, neg_il, item_list_emb)
        pos_score = self.forward(interaction)
        neg_score = torch.squeeze(neg_score)
        loss = - self.loss(pos_score, neg_score)
        return loss

    def predict(self, interaction):
        score = self.forward(interaction)
        return score

    def full_sort_predict(self, interaction):
        pass
