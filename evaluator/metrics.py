import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error
)

"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""

#    TopK Metrics    #


def hit(pos_index, pos_len):
    """Hit(also known as hit ratio at N) is a way of calculating how many “hits” you have in an n-sized list of ranked items.

    url:https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

    """
    result = np.cumsum(pos_index, axis=1)
    return (result > 0).astype(int)


def mrr(pos_index, pos_len):
    """The MRR (also known as mean reciprocal rank) is a statistic measure for evaluating any process that produces a list
    of possible responses to a sample of queries, ordered by probability of correctness.

    url:https://en.wikipedia.org/wiki/Mean_reciprocal_ranks

    """

    idxs = pos_index.argmax(axis=1)
    result = np.zeros_like(pos_index, dtype=np.float)
    for row, idx in enumerate(idxs):
        if pos_index[row, idx] > 0:
            result[row, idx:] = 1 / (idx + 1)
        else:
            result[row, idx:] = 0
    return result


def map_(truth_ranks, pos_len):
    raise NotImplementedError


def recall(pos_index, pos_len):
    """recall (also known as sensitivity) is the fraction of the total amount of relevant instances that were actually
    retrieved

    url:https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    """
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def ndcg(pos_index, pos_len):
    """NDCG (also known as normalized discounted cumulative gain) is a measure of ranking quality. Through normalizing the
    score, users and their recommendation list results in the whole test set can be evaluated.

    url:https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    """

    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index)
    iranks[:, :] = np.arange(1, pos_index.shape[1]+1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index)
    ranks[:, :] = np.arange(1, pos_index.shape[1]+1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result


def precision(pos_index, pos_len):
    """precision (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances

    url:https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    """
    return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1]+1)


#    CTR Metrics    #


def auc(trues, preds):
    """AUC (also known as Area Under Curve) is used to evaluate the two-class model,
     referring to the area under the ROC curve

    url:https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    Note:
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users.

    """
    return roc_auc_score(trues, preds)


# Loss based Metrics #


def mae(trues, preds):
    """[summary]

    url:
    """
    return mean_absolute_error(trues, preds)


def rmse(trues, preds):
    """[summary]

    url:
    """
    return np.sqrt(mean_squared_error(trues, preds))


# Item based Metrics #


def coverage(n_items, ):
    pass


def gini_index():
    pass


def shannon_entropy():
    pass


def diversity():
    pass


metrics_dict = {
    'ndcg': ndcg,
    'hit': hit,
    'precision': precision,
    'map': map_,
    'recall': recall,
    'mrr': mrr,
    'rmse': rmse,
    'mae': mae
}