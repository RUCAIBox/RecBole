import numpy as np
import pandas as pd
import sys
##    TopK Metrics    ##

def hit(data):  
    return any(data > 0)

def mrr(data):
    tmp_x = data[data > 0]
    if not tmp_x.empty:
        return (1 / tmp_x).sum() / data.shape[0]
    return 0

def recall(data):
    return (data > 0).sum() / data.shape[0]

def ndcg(data):
    pass

def precision(data, k):
    return (data > 0).sum() / k

def auc(data, neg_ratio):
    if neg_ratio == 0:
        return 1
    pos_num = (data > 0).sum()
    if pos_num == 0:
        return 0
    pos_ranksum = (pos_num * (neg_ratio + 1) + 1 - data[data > 0]).sum()
    neg_num = pos_num * neg_ratio
    return (pos_ranksum - pos_num * (pos_num + 1) / 2) / (pos_num * neg_num)

## Loss based Metrics ##

def mae(data):
    pass

def rmse(data):
    pass

## Item based Metrics ##

def coverage(n_items, ):
    pass

def gini_index():
    pass

def shannon_entropy():
    pass

def diversity():
    pass