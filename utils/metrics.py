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

def precision(data):
    pass

def auc(data):
    pass

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