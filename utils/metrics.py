import numpy as np
import pandas as pd
import sys
##    TopK Metrics    ##


def hit(truth_ranks, k):  
    """Hit(also known as hit ratio at N) is a way of calculating how many “hits” you have in an n-sized list of ranked items.

    url:https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

    """
    return int(any(truth_ranks <= k))


def mrr(truth_ranks, k):
    """The MRR (also known as mean reciprocal rank) is a statistic measure for evaluating any process that produces a list
    of possible responses to a sample of queries, ordered by probability of correctness. 

    url:https://en.wikipedia.org/wiki/Mean_reciprocal_ranks

    """
    truth_ranks_at_k = truth_ranks[truth_ranks <= k]
    if truth_ranks_at_k.shape[0] > 0:
        return (1 / truth_ranks_at_k.min())
    return 0


def recall(truth_ranks, k,  pos_num):
    """recall (also known as sensitivity) is the fraction of the total amount of relevant instances that were actually 
    retrieved.  

    url:https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    """
    return (truth_ranks <= k).sum() / pos_num


def ndcg(truth_ranks, k, pos_num):
    """NDCG (also known as normalized discounted cumulative gain) is a measure of ranking quality. Through normalizing the
    score, users and their recommendation list results in the whole test set can be evaluated.

    url:https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    """
    truth_ranks_at_k = truth_ranks[truth_ranks <= k]
    idcg_len = min(k, pos_num)
    idcg = np.sum(1.0 / np.log2(np.arange(2, idcg_len + 2)))
    dcg = np.sum(1.0 / np.log2(truth_ranks_at_k + 1))
    result = dcg / idcg
    return result

def precision(truth_ranks, k):
    """precision (also called positive predictive value) is the fraction of relevant instances among the retrieved 
    instances

    url:https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    """
    return (truth_ranks <= k).sum() / k

def auc(rank, label, k=None):
    """AUC (also known as Area Under Curve) is used to evaluate the two-class model, referring to the area under the ROC curve

    url:https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    """ 
    pos_num = label.sum()
    neg_num = (~label).sum()
    if pos_num == 0:
        return 0
    if neg_num == 0:
        return 1
    pos_ranksum = ((rank.shape[0]) + 1 - rank[label]).sum()

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


    