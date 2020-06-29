# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Time    :   2020/06/28 11:32:23
@Author  :   tsotfsk
@Version :   1.0
@Contact :   tsotfsk@outlook.com
'''

# here put the import lib

import numpy as np
import pandas as pd
import sys

# @profile
def hit(data, k):
    def get_hit(x):
        return any(~x.isnull())
    num_users = data['user_id'].nunique()
    topk_df = data[data['rank'] <= k]
    grouped = topk_df.groupby('user_id')['score'].apply(get_hit)
    return grouped.sum() / num_users

# @profile
def mrr(data, k):
    def get_mrr(x):
        tmp_x = x[x > 0]
        if not tmp_x.empty:
            return (1 / tmp_x).sum() / x.shape[0]
        return 0
    num_users = data['user_id'].nunique()
    topk_df = data[data['rank'] <= k]
    grouped = topk_df.groupby('user_id')['rank'].apply(get_mrr)
    return grouped.sum() / num_users

# @profile
def recall(data, k):
    def get_recall(x):
        return 1 - x.isnull().sum() / x.shape[0]
    num_users = data['user_id'].nunique()
    topk_df = data[data['rank'] <= k]
    grouped = topk_df.groupby('user_id')['score'].apply(get_recall)
    return grouped.sum() / num_users
