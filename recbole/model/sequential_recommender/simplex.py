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
import torch.nn.functional as F

import numpy as np

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
        self.UI_aggregation_weight=config['UI_aggregation_weight']
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
    #pos_item_seq:[n,MAX_ITEM_LIST_LENGTH]，类型tensor。
    # pos_seq_len，[n]pos_item_seq有效的长度，类型tensor
    # neg_item_seq:[n,neg_seq_len]，类型tensor
    # forward函数的作用是为了获得loss
    def forward(self, user, pos_item, pos_item_seq, pos_seq_len, neg_item_seq):

        self.full_sort_predict(user,pos_item_seq,pos_seq_len)
        # self.predict(user,pos_item_seq,pos_seq_len,pos_item)

        # [n,embedding_size]
        user_e=self._get_emb_normalization(user,self.user_emb)
        #[n,embedding_size]
        pos_item_e=self._get_emb_normalization(pos_item,self.item_emb)
        #[n,MAX_ITEM_LIST_LENGTH,embedding_size]
        pos_item_seq_e=self._get_emb_normalization(pos_item_seq,self.item_emb)
        #[n,neg_seq_len,embedding_size]
        neg_item_seq_e=self._get_emb_normalization(neg_item_seq,self.item_emb)
        
        # 获得UI_aggregation_e，大小是[n,embedding_size]
        UI_aggregation_e=self._get_UI_aggregation(user_e,pos_item_seq_e,pos_seq_len)

        UI_pos_cos=self._get_cos(UI_aggregation_e,pos_item_e.unsqueeze(1))
        UI_neg_seq_cos=self._get_cos(UI_aggregation_e,neg_item_seq_e)

        pos_score=1-UI_pos_cos
        temp=UI_neg_seq_cos-self.CCL_margin
        temp=torch.maximum(torch.zeros(1),temp)
        temp=temp.sum(1).unsqueeze(1)
        neg_score=self.CCL_weight/self.neg_seq_len*temp

        CCL_loss=(pos_score+neg_score).mean()
        return CCL_loss
    
    # 返回归一化特征向量
    # id长度为一维，embedding_layer为对应嵌入层
    def _get_emb_normalization(self,id,embedding_layer):
        id_e = embedding_layer(id)
        id_e=F.normalize(id_e,dim=id_e.dim()-1)
        return id_e

    # 获得所有的SimpleX中user和项目序列的结合向量
    # user_e：[n,embedding_size]类型tesor，n是用户数，每一行代表第i个用户的特征向量
    # pos_item_seq_e：[n,MAX_ITEM_LIST_LENGTH,embedding_size]类型tensor
    # pos_seq_len:[n]，类型tensor，代表pos_item_seq_e：在第二维的有效长度
    # 输出：[n,embedding_size]类型tensor，第i行是user_i和项目序列的结合向量
    def _get_UI_aggregation(self,user_e,pos_item_seq_e,pos_seq_len):
        # 获得UI_aggregation_e，大小是[n,embedding_size]
        UI_aggregation_e=[0]*len(user_e)
        for i in range(len(user_e)):
            pos_item_mean=pos_item_seq_e[i][0:pos_seq_len[i]].mean(dim=0)
            g=self.UI_aggregation_weight
            UI_aggregation_e[i]=g*user_e[i]+(1-g)*pos_item_mean
        UI_aggregation_e=torch.stack(UI_aggregation_e)
        return UI_aggregation_e
    
    # 获得SimpleX的余弦值
    # UI_aggregation_e：[n,embedding_size]类型tesor，n是用户数量，每一行是user和项目序列的结合向量
    # item_e：[n,m,embedding_size]类型tensor，m代表项目数量，每一行是项目的特征向量
    # 输出：[n,m]类型tensor，[i,j]代表第i个用户对第j个项目的余弦值
    def _get_cos(self,UI_aggregation_e,item_e):
        # 计划使用归一化后进行点乘即可
        UI_aggregation_e=F.normalize(UI_aggregation_e,dim=1)
        # [n,embedding_size,1]
        UI_aggregation_e=UI_aggregation_e.unsqueeze(2)
        item_e=F.normalize(item_e,dim=2)
        UI_cos=torch.matmul(item_e,UI_aggregation_e)
        return UI_cos.squeeze(2)

    def calculate_loss(self, interaction):
        #数据处理
        # 想使用SimpleX，首先一个用户要对应一批自己的商品时间序列，一个pos item和一批neg_item序列
        #基于RecBole框架，我们将数据按用户进行区分，一个用户有neg_seq_len条数据
        # 对于商品时间序列，和pos item，从多条数据中使用随机数随机抽一个出来
        #对于neg item序列，我们将多条数据的neg item聚合起来成为一个序列
        # 由于RecBole的特性，是比较容易获得以上属性的
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        pos_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        # 获得neg item序列
        neg_items_seq=neg_items.reshape((self.neg_seq_len,-1))
        neg_items_seq=torch.transpose(neg_items_seq, 0, 1)

        user_number=int(len(user)/self.neg_seq_len)
        rand=np.random.randint(self.neg_seq_len,size=user_number)
        # select_index是第i个用户选中的那条数据的下标
        select_index=[i+rand[i]*user_number for i in range(user_number)]
        # 获得user，商品时间序列，和pos item，pos_seq_len
        user=user[select_index]
        pos_item_seq=item_seq[select_index]
        pos_items=pos_items[select_index]
        pos_seq_len=pos_seq_len[select_index]

        loss = self.forward(user,pos_items,pos_item_seq,pos_seq_len,neg_items_seq)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
    # def predict(self, user,item_seq,item_seq_len,test_item):

        # [n,embedding_size]
        user_e=self._get_emb_normalization(user,self.user_emb)
        #[n,embedding_size]
        test_item_e=self._get_emb_normalization(test_item,self.item_emb)
        #[n,MAX_ITEM_LIST_LENGTH,embedding_size]
        item_seq_e=self._get_emb_normalization(item_seq,self.item_emb)
        
        # 获得UI_aggregation_e，大小是[n,embedding_size]
        UI_aggregation_e=self._get_UI_aggregation(user_e,item_seq_e,item_seq_len)

        UI_cos=self._get_cos(UI_aggregation_e,test_item_e.unsqueeze(1))
        return UI_cos

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
    # def full_sort_predict(self, user,item_seq,item_seq_len):

        # [n,embedding_size]
        user_e=self._get_emb_normalization(user,self.user_emb)
        #[n,MAX_ITEM_LIST_LENGTH,embedding_size]
        item_seq_e=self._get_emb_normalization(item_seq,self.item_emb)
        
        # 获得UI_aggregation_e，大小是[n,embedding_size]
        UI_aggregation_e=self._get_UI_aggregation(user_e,item_seq_e,item_seq_len)

        UI_aggregation_e=F.normalize(UI_aggregation_e,dim=1)
        all_item_emb=self.item_emb.weight
        all_item_emb=F.normalize(all_item_emb,dim=1)
        all_item_emb=torch.transpose(all_item_emb, 0, 1)
        UI_cos=torch.matmul(UI_aggregation_e,all_item_emb)
        return UI_cos

