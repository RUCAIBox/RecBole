# -*- coding: utf-8 -*-
# @Time   : 2022/3/25 13:38
# @Author : HaoJun Qin
# @Email  : 18697951462@qq.com

r"""
SimpleX
################################################

Reference:
    Kelong Mao et al. "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." in CIKM 2021.

"""
import torch
from torch import nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class SimpleX(SequentialRecommender):
    # 更改
    r"""The FPMC model is mainly used in the recommendation system to predict the possibility of
    unknown items arousing user interest, and to discharge the item recommendation list.

    Note:

        In order that the generation method we used is common to other sequential models,
        We set the size of the basket mentioned in the paper equal to 1.
        For comparison with other models, the loss function used is BPR.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimpleX, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.CCL_margin=config['CCL_margin']
        self.CCL_weight=config['CCL_weight']
        self.user_item_aggregation_weight=config['user_item_aggregation_weight']
        # 一个用户对应的负采样序列长度
        self.neg_seq_len=config['neg_sampling']['uniform']

        # load dataset info
        self.n_users = dataset.user_num

        # define layers and loss
        # user embedding matrix
        self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
        # item embedding matrix
        self.item_emb = nn.Embedding(self.n_items, self.embedding_size)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    #user:[n]，类型tensor。pos_item:[n]，类型tensor
    #pos_item_seq:[n,MAX_ITEM_LIST_LENGTH]，类型tensor。pos_seq_len，有效的长度，类型tensor
    # neg_item_seq:[n,neg_seq_len]，类型tensor
    # 目的是为了获得loss
    def forward(self, user, pos_item, pos_item_seq, pos_seq_len, neg_item_seq):
        item_last_click_index = item_seq_len - 1
        item_last_click = torch.gather(item_seq, dim=1, index=item_last_click_index.unsqueeze(1))
        item_seq_emb = self.LI_emb(item_last_click)  # [b,1,emb]

        user_emb = self.UI_emb(user)
        user_emb = torch.unsqueeze(user_emb, dim=1)  # [b,1,emb]

        iu_emb = self.IU_emb(next_item)
        iu_emb = torch.unsqueeze(iu_emb, dim=1)  # [b,n,emb] in here n = 1

        il_emb = self.IL_emb(next_item)
        il_emb = torch.unsqueeze(il_emb, dim=1)  # [b,n,emb] in here n = 1

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
        #数据处理
        # 想使用SimpleX，首先一个用户要对应一批自己的商品时间序列，一个pos item和一批neg_item序列
        #基于RecBole框架，我们将数据按用户进行区分，一个用户有neg_seq_len条数据。
        # 对于商品时间序列，和pos item，从多条数据中使用随机数随机抽一个出来
        #对于neg item序列，我们将多条数据的neg item聚合起来成为一个序列
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        pos_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        # 获得neg item序列
        neg_items=torch.reshape(neg_items,(self.neg_seq_len,-1))
        neg_items=torch.transpose(neg_items, 0, 1)

        # 当前训练用户人数
        user_number=int(len(user)/self.neg_seq_len)
        rand=torch.rand(user_number).cpu().numpy()
        rand=(rand*self.neg_seq_len).astype(int)
        # select_index是第i个用户选中的那条数据的下标
        select_index=[i+rand[i]*user_number for i in range(user_number)]
        # 获得user，商品时间序列，和pos item，pos_seq_len
        user=user[select_index]
        item_seq=item_seq[select_index]
        pos_items=pos_items[select_index]
        pos_seq_len=pos_seq_len[select_index]


        # # 按用户分组
        # user_group_dict={}
        # user_numpy=user.cpu().numpy()
        # for i in range(len(user_numpy)):
        #     if user_numpy[i] not in user_group_dict:
        #         user_group_dict[user_numpy[i]]=[]
        #     user_group_dict[user_numpy[i]].append(i)
        # # 对每一个user，抽一条数据获得商品时间序列，和pos item
        # dict_index=torch.rand(len(user_group_dict)).cpu().numpy()
        # dict_index=(dict_index*self.neg_seq_len).astype(int)

        loss = self.forward(user,pos_items,item_seq,pos_seq_len,neg_items)
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
        mf = torch.matmul(user_emb, all_iu_emb.transpose(0, 1))
        all_il_emb = self.IL_emb.weight

        item_last_click_index = item_seq_len - 1
        item_last_click = torch.gather(item_seq, dim=1, index=item_last_click_index.unsqueeze(1))
        item_seq_emb = self.LI_emb(item_last_click)  # [b,1,emb]
        fmc = torch.matmul(item_seq_emb, all_il_emb.transpose(0, 1))
        fmc = torch.squeeze(fmc, dim=1)
        score = mf + fmc
        return score
