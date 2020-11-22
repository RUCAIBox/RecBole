# -*- coding: utf-8 -*-
# @Time    : 2020/11/22 12:08
# @Author  : Shao Weiqi
# @Email   : shaoweiqi@ruc.edu.cn

r"""
HRM
################################################

Reference:
    Pengfei Wang et al. "Learning Hierarchical Representation Model for Next Basket Recommendation." in SIGIR 2015.


"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_,constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class HRM(SequentialRecommender):
    r"""
     HRM can well capture both sequential behavior and users’ general taste by involving transaction and user representations in prediction
     HRM user max- & average- pooling as a good helper
    """

    def __init__(self,config,dataset):
        super(HRM, self).__init__(config,dataset)

        # load the dataset information
        self.n_user=dataset.num(self.USER_ID)
        self.device=config["device"]

        # load the parameters information
        self.embedding_size=config["embedding_size"]
        self.pooling_type_layer_1=config["pooling_type_layer_1"]
        self.pooling_type_layer_2=config["pooling_type_layer_2"]
        self.high_order=config["high_order"]
        self.reg_weight=config["reg_weight"]

        # define the layers and loss type
        self.item_embedding=nn.Embedding(self.n_items,self.embedding_size,padding_idx=0)
        self.user_embedding=nn.Embedding(self.n_user,self.embedding_size)

        self.loss_type = config['loss_type']
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameters of the model
        self.apply(self._init_weights)


    def inverse_seq_item(self,seq_item):

        seq_item=seq_item.numpy()
        for idx,x in enumerate(seq_item):
            seq_item[idx]=x[::-1]
        seq_item=torch.LongTensor(seq_item).to(self.device)

        return seq_item


    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)


    def forward(self,seq_item,user,seq_item_len):

        seq_item=self.inverse_seq_item(seq_item)

        # seq_item=self.inverse_seq_item(seq_item)
        seq_item_embedding=self.item_embedding(seq_item)
        # batch_size * seq_len * embedding_size
        high_order_item_embedding=seq_item_embedding[:,-self.high_order:,:]
        # batch_size * high_order * embedding_size
        user_embedding=self.user_embedding(user)
        # batch_size * embedding_size

        # layer 1
        if self.pooling_type_layer_1=="max":
            high_order_item_embedding=torch.max(high_order_item_embedding,dim=1)[0]
            # batch_size * embedding_size
        else:
            for idx,x in enumerate(seq_item_len):
                if x>self.high_order:
                    seq_item_len[idx]=self.high_order
                    # batch_size
            high_order_item_embedding=torch.sum(high_order_item_embedding,dim=1)
            high_order_item_embedding=torch.div(high_order_item_embedding,seq_item_len.unsqueeze(1))
            # batch_size * embedding_size
        hybrid_user_embedding=torch.cat([user_embedding,high_order_item_embedding],dim=1)
        # batch_size * 2_mul_embedding_size

        # layer 2
        if self.pooling_type_layer_2=="max":
            hybrid_user_embedding=torch.max(hybrid_user_embedding,dim=1,keepdim=True)[0]
            # batch_size * 1
        else:
            hybrid_user_embedding=torch.mean(hybrid_user_embedding,dim=1,keepdim=True)
            # batch_size * 1

        return hybrid_user_embedding


    def reg_loss(self,user_embedding,item_embedding,seq_item_embedding):

        reg_2=self.reg_weight
        loss_2=reg_2*torch.norm(user_embedding,p=2)+reg_2*torch.norm(item_embedding,p=2)+reg_2*torch.norm(seq_item_embedding,p=2)

        return loss_2


    def calculate_loss(self, interaction):

        seq_item=interaction[self.ITEM_SEQ]
        seq_item_len=interaction[self.ITEM_SEQ_LEN]
        user=interaction[self.USER_ID]
        user_embedding=self.user_embedding(user)
        seq_output=self.forward(seq_item,user,seq_item_len)
        pos_items=interaction[self.POS_ITEM_ID]
        seq_item_embedding=self.item_embedding(seq_item)[:,-self.high_order:,:]
        pos_items_emb = self.item_embedding(pos_items)
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss + self.reg_loss(user_embedding,pos_items_emb,seq_item_embedding)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            seq_output=seq_output.repeat(1,self.embedding_size)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss+self.reg_loss(user_embedding,pos_items_emb,seq_item_embedding)


    def predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user,seq_item_len)
        seq_output = seq_output.repeat(1, self.embedding_size)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores


    def full_sort_predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user,seq_item_len)
        seq_output = seq_output.repeat(1, self.embedding_size)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores