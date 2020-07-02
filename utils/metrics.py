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
    tmp_data = data.copy()
    def get_hit(x):
        return any(x > 0)
    num_users = tmp_data['user_id'].nunique()
    mask = tmp_data['rank'] > k
    tmp_data.loc[mask, 'rank'] = -1
    grouped = tmp_data.groupby('user_id')['rank'].apply(get_hit)
    return grouped.sum() / num_users

# @profile
def mrr(data, k):
    tmp_data = data.copy()
    def get_mrr(x):
        tmp_x = x[x > 0]
        if not tmp_x.empty:
            return (1 / tmp_x).sum() / x.shape[0]
        return 0
    num_users = tmp_data['user_id'].nunique()
    mask = tmp_data['rank'] > k
    tmp_data.loc[mask, 'rank'] = -1
    grouped = tmp_data.groupby('user_id')['rank'].apply(get_mrr)
    return grouped.sum() / num_users

# @profile
def recall(data, k):
    tmp_data = data.copy()
    def get_recall(x):
        return (x > 0).sum() / x.shape[0]
    num_users = tmp_data['user_id'].nunique()
    mask = tmp_data['rank'] > k
    tmp_data.loc[mask, 'rank'] = -1
    grouped = tmp_data.groupby('user_id')['rank'].apply(get_recall)
    return grouped.sum() / num_users
