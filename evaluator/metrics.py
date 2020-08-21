# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/12   2020/08/21
# @Author  :   Kaiyuan Li   Zhichao Feng
# @email   :   tsotfsk@outlook.com  fzcbupt@gmail.com

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

from .utils import _binary_clf_curve

#    TopK Metrics    #


def hit(pos_index, pos_len):
    r"""Hit(also known as hit ratio at N) is a way of calculating how many 'hits' you have in an n-sized list of ranked items.

    url:https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

    $$
        \mathrm {HR@K} =\frac{Number of Hits @K}{|GT|}
    $$

    HR is the number of users with a positive sample in the recommendation list.GT is the total number of samples in the test set.
    """
    result = np.cumsum(pos_index, axis=1)
    return (result > 0).astype(int)


def mrr(pos_index, pos_len):
    r"""The MRR (also known as mean reciprocal rank) is a statistic measure for evaluating any process that produces a list
    of possible responses to a sample of queries, ordered by probability of correctness.

    url:https://en.wikipedia.org/wiki/Mean_reciprocal_ranks

    $$
        \mathrm {MRR} = \frac{1}{|{U}|} \sum_{i=1}^{|{U}|} \frac{1}{rank_i}
    $$

    ${U}$ is the number of users, $rank_i$ is the rank of the first item in the recommendation list
    in the test set results for user ${i}$.
    """

    idxs = pos_index.argmax(axis=1)
    result = np.zeros_like(pos_index, dtype=np.float)
    for row, idx in enumerate(idxs):
        if pos_index[row, idx] > 0:
            result[row, idx:] = 1 / (idx + 1)
        else:
            result[row, idx:] = 0
    return result


def map_(pos_index, pos_len):
    r"""map (also known as Mean Average Precision) The MAP is meant to calculate Avg. Precision for the relevant items.

    Note:

            In this case the normalization factor used is $ \frac{1}{\min (m,N)} $, which prevents your AP score from
            being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones.

    url:http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

    $$
        \begin{align*}
        \mathrm{AP@N} &= \frac{1}{\mathrm{min}(m,N)}\sum_{k=1}^N P(k) \cdot rel(k) \\
        \mathrm{MAP@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{AP@N})_u
        \end{align*}
    $$
    """
    pre = precision(pos_index, pos_len)
    sum_pre = np.cumsum(pre * pos_index.astype(np.float), axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
    result = np.zeros_like(pos_index, dtype=np.float)
    for row, lens in enumerate(actual_len):
        ranges = np.arange(1, pos_index.shape[1]+1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result


def recall(pos_index, pos_len):
    r"""recall (also known as sensitivity) is the fraction of the total amount of relevant instances that were actually
    retrieved

    url:https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    $$
        \mathrm {Recall@K} = \frac{|Rel_u\cap Rec_u|}{Rel_u}
    $$

    ${Rel_u}$ is the set of items relavent to user ${U}$, ${Rec_u}$ is the top K items recommended to users.
    We obtain the result by calculating the average ${Recall@K}$ of each user.
    """
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def ndcg(pos_index, pos_len):
    r"""NDCG (also known as normalized discounted cumulative gain) is a measure of ranking quality. Through normalizing the
    score, users and their recommendation list results in the whole test set can be evaluated.

    url:https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG


    \begin{gather}
        \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
        \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
        \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
        \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in u^{te}NDCG_u@K}}{|u^{te}|}
    \end{gather}

    ${K}$ stands for recommending ${K}$ items.And the ${rel_i}$ is the relevance of the item in position ${i}$ in the
    recommendation list.$2^{rel_i}$ equals to 1 if the item hits otherwise 0.${U^{te}}$ is for all users in the test set.
    """

    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result


def precision(pos_index, pos_len):
    r"""precision (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances

    url:https://en.wikipedia.org/wiki/Precision_and_recall#Precision


    $$
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}
    $$

    ${Rel_u}$ is the set of items relavent to user ${U}$, ${Rec_u}$ is the top K items recommended to users.
    We obtain the result by calculating the average ${Precision@K}$ of each user.

    """
    return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)


#    CTR Metrics    #

def auc(trues, preds):
    r"""AUC (also known as Area Under Curve) is used to evaluate the two-class model,
     referring to the area under the ROC curve

    url:https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    Note:
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users.

    $$
        \mathrm {AUC} = \frac{\sum\limits_{i=1}^M rank_{i}
        - {{M} \times {(M+1)}}} {{M} \times {N}}
    $$

    M is the number of positive samples.N is the number of negative samples.${rank_i}$ is the rank of the ith positive sample.
    """
    fps, tps = _binary_clf_curve(trues, preds)
    optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
    fps = np.r_[0, fps[optimal_idxs]]
    tps = np.r_[0, tps[optimal_idxs]]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return sk_auc(fpr, tpr)


# Loss based Metrics #

def mae(trues, preds):
    r"""Mean absolute error regression loss

    url:https://en.wikipedia.org/wiki/Mean_absolute_error

    $$
        \mathrm{MAE}=\frac{1}{|{T}|} \sum_{(u, i) \in {T}}\left|\hat{r}_{u i}-r_{u i}\right|
    $$

    ${T}$ is the test set, $\hat{r}_{u i}$ is the score predicted by the model, and $r_{u i}$ the actual score of the test set.

    """

    return mean_absolute_error(trues, preds)


def rmse(trues, preds):
    r"""Mean std error regression loss

    url:https://en.wikipedia.org/wiki/Root-mean-square_deviation

    $$
        \mathrm{RMSE} = \sqrt{\frac{1}{|{T}|} \sum_{(u, i) \in {T}}(\hat{r}_{u i}-r_{u i})^{2}}
    $$

    ${T}$ is the test set, $\hat{r}_{u i}$ is the score predicted by the model, and $r_{u i}$ the actual score of the test set.

    """
    return np.sqrt(mean_squared_error(trues, preds))


def log_loss_(trues, preds):
    r"""Log loss, aka logistic loss or cross-entropy loss

    url:http://wiki.fast.ai/index.php/Log_Loss

    $$
        -\log {P(yt|yp)} = -(({yt}\ \log{yp}) + {(1-yt)}\ \log{(1 - yp)})
    $$

    For a single sample, yt is true label in {0,1}.
    yp is the estimated probability that yt = 1.

    """
    eps = 1e-15
    preds = np.float64(preds)
    preds = np.clip(preds, eps, 1 - eps)
    loss = np.sum(- trues * np.log(preds) - (1 - trues) * np.log(1 - preds))

    return loss / len(preds)

# Item based Metrics #


def coverage():
    raise NotImplementedError


def gini_index():
    raise NotImplementedError


def shannon_entropy():
    raise NotImplementedError


def diversity():
    raise NotImplementedError


"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
metrics_dict = {
    'ndcg': ndcg,
    'hit': hit,
    'precision': precision,
    'map': map_,
    'recall': recall,
    'mrr': mrr,
    'rmse': rmse,
    'mae': mae,
    'logloss': log_loss_,
    'auc': auc,
    'map': map_
}
