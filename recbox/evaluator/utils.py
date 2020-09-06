# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/09
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

from enum import Enum
import numpy as np


# class TOPK_ARGS(Enum):
#     POS_INDEX = 0
#     POS_LEN = 1

#     NDCG = (POS_INDEX, POS_LEN)
#     MAP = (POS_INDEX, POS_LEN)
#     RECALL = (POS_INDEX, POS_LEN)
#     MRR = (POS_INDEX)
#     HIT = (POS_INDEX)
#     PRECISION = (POS_INDEX)


class TOPK_METRICS(Enum):
    NDCG = 'ndcg'
    MRR = 'mrr'
    MAP = 'map'
    HIT = 'hit'
    RECALL = 'recall'
    PRECISION = 'precision'


class LOSS_METRICS(Enum):
    MAE = 'mae'
    RMSE = 'rmse'
    LOGLOSS = 'logloss'
    AUC = 'auc'


class ITEM_METRIC(Enum):
    pass


def trunc(scores, method):
    """Round the scores by using the given method

    Args:
        scores (np.ndarray): scores
        method (str): one of ['ceil', 'floor', 'around']

    Raises:
        NotImplementedError: method error

    Returns:
        (np.ndarray): processed scores
    """

    try:
        cut_method = getattr(np, method)
    except NotImplementedError as e:
        raise NotImplementedError("module 'numpy' has no fuction named '{}'".format(method))
    scores = cut_method(scores)
    return scores


def cutoff(scores, threshold):
    """cut of the scores based on threshold

    Args:
        scores (np.ndarray): scores
        threshold (float): between 0 and 1

    Returns:
        (np.ndarray): processed scores
    """
    return np.where(scores > threshold, 1, 0)


def _binary_clf_curve(trues, preds):
    """Calculate true and false positives per binary classification threshold

    Args:
        trues (numpy.ndarray): the true scores' list
        preds (numpy.ndarray): the predict scores' list

    Returns:
        fps (np.ndarray): A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]
        preds (numpy.ndarray): An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i].

    Note:
        To improve efficiency, we referred to the source code(which is available at sklearn.metrics.roc_curve)
        in SkLearn and made some optimizations.
    """
    trues = (trues == 1)

    desc_idxs = np.argsort(preds)[::-1]
    preds = preds[desc_idxs]
    trues = trues[desc_idxs]

    unique_val_idxs = np.where(np.diff(preds))[0]
    threshold_idxs = np.r_[unique_val_idxs, trues.size - 1]

    tps = np.cumsum(trues)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps
