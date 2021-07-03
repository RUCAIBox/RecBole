# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/12, 2021/6/25, 2020/9/16, 2021/7/2
# @Author  :   Kaiyuan Li, Zhichao Feng, Xingyu Pan, Zihan Lin
# @email   :   tsotfsk@outlook.com, fzcbupt@gmail.com, panxy@ruc.edu.cn, zhlin@ruc.edu.cn

"""
recbole.evaluator.metrics
############################
"""

from logging import getLogger

import numpy as np
from collections import Counter
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import mean_absolute_error, mean_squared_error

from recbole.evaluator.utils import _binary_clf_curve
from recbole.evaluator.base_metric import TopkMetric, LossMetric


#    TopK Metrics    #

class Hit(TopkMetric):
    r"""Hit_ (also known as hit ratio at :math:`N`) is a way of calculating how many 'hits' you have
    in an n-sized list of ranked items.
    .. _Hit: https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870
    .. math::
        \mathrm {HR@K} =\frac{Number \space of \space Hits @K}{|GT|}
    :math:`HR` is the number of users with a positive sample in the recommendation list.
    :math:`GT` is the total number of samples in the test set.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result('hit', result)
        return metric_dict

    def metric_info(self, pos_index):
        result = np.cumsum(pos_index, axis=1)
        return (result > 0).astype(int)


class MRR(TopkMetric):
    r"""The MRR_ (also known as mean reciprocal rank) is a statistic measure for evaluating any process
   that produces a list of possible responses to a sample of queries, ordered by probability of correctness.
   .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
   .. math::
       \mathrm {MRR} = \frac{1}{|{U}|} \sum_{i=1}^{|{U}|} \frac{1}{rank_i}
   :math:`U` is the number of users, :math:`rank_i` is the rank of the first item in the recommendation list
   in the test set results for user :math:`i`.
   """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result('mrr', result)
        return metric_dict

    def metric_info(self, pos_index):
        idxs = pos_index.argmax(axis=1)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, idx in enumerate(idxs):
            if pos_index[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result


class MAP(TopkMetric):
    r"""MAP_ (also known as Mean Average Precision) The MAP is meant to calculate Avg. Precision for the relevant items.
    Note:
        In this case the normalization factor used is :math:`\frac{1}{\min (m,N)}`, which prevents your AP score from
        being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones.
    .. _MAP: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms
    .. math::
       \begin{align*}
       \mathrm{AP@N} &= \frac{1}{\mathrm{min}(m,N)}\sum_{k=1}^N P(k) \cdot rel(k) \\
       \mathrm{MAP@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{AP@N})_u
       \end{align*}
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result('map', result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
        sum_pre = np.cumsum(pre * pos_index.astype(np.float), axis=1)
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, pos_index.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result


class Recall(TopkMetric):
    r"""Recall_ (also known as sensitivity) is the fraction of the total amount of relevant instances
    that were actually retrieved
    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall
    .. math::
       \mathrm {Recall@K} = \frac{|Rel_u\cap Rec_u|}{Rel_u}
    :math:`Rel_u` is the set of items relevant to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Recall@K` of each user.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result('recall', result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


class NDCG(TopkMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.
    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in U^{te}NDCG_u@K}}{|U^{te}|}
        \end{gather}
    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`{rel_i}` equals to 1 if the item is ground truth otherwise 0.
    :math:`U^{te}` stands for all users in the test set.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result('ndcg', result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
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


class Precision(TopkMetric):
    r"""Precision_ (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances
    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision
    .. math::
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}
    :math:`Rel_u` is the set of items relevant to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Precision@K` of each user.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result('precision', result)
        return metric_dict

    def metric_info(self, pos_index):
        return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)


# CTR Metrics

class GAUC(object):
    r"""GAUC_ (also known as Group Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve grouped by user.
    .. _GAUC: https://dl.acm.org/doi/10.1145/3219819.3219823
    Note:
        It calculates the AUC score of each user, and finally obtains GAUC by weighting the user AUC.
        It is also not limited to k. Due to our padding for `scores_tensor` in `RankEvaluator` with
        `-np.inf`, the padding value will influence the ranks of origin items. Therefore, we use
        descending sort here and make an identity transformation  to the formula of `AUC`, which is
        shown in `auc_` function. For readability, we didn't do simplification in the code.
    .. math::
        \mathrm {GAUC} = \frac {{{M} \times {(M+N+1)} - \frac{M \times (M+1)}{2}} -
        \sum\limits_{i=1}^M rank_{i}} {{M} \times {N}}
    :math:`M` is the number of positive samples.
    :math:`N` is the number of negative samples.
    :math:`rank_i` is the descending rank of the ith positive sample.
    """

    def __init__(self, config):
        self.decimal_place = config['metric_decimal_place']

    def calculate_metric(self, dataobject):
        meanrank = dataobject.get('rec.meanrank')
        meanrank = meanrank.numpy()
        pos_rank_sum, user_len_list, pos_len_list = np.split(meanrank, 3, axis=1)
        user_len_list, pos_len_list = user_len_list.squeeze(), pos_len_list.squeeze()
        result = self.metric_info(pos_rank_sum, user_len_list, pos_len_list)
        return {'gauc': round(result, self.decimal_place)}

    def metric_info(self, pos_rank_sum, user_len_list, pos_len_list):
        neg_len_list = user_len_list - pos_len_list
        # check positive and negative samples
        any_without_pos = np.any(pos_len_list == 0)
        any_without_neg = np.any(neg_len_list == 0)
        non_zero_idx = np.full(len(user_len_list), True, dtype=np.bool)
        if any_without_pos:
            logger = getLogger()
            logger.warning(
                "No positive samples in some users, "
                "true positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= (pos_len_list != 0)
        if any_without_neg:
            logger = getLogger()
            logger.warning(
                "No negative samples in some users, "
                "false positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= (neg_len_list != 0)
        if any_without_pos or any_without_neg:
            item_list = user_len_list, neg_len_list, pos_len_list, pos_rank_sum
            user_len_list, neg_len_list, pos_len_list, pos_rank_sum = \
                map(lambda x: x[non_zero_idx], item_list)

        pair_num = (user_len_list + 1) * pos_len_list - pos_len_list * (pos_len_list + 1) / 2 - np.squeeze(pos_rank_sum)
        user_auc = pair_num / (neg_len_list * pos_len_list)

        result = (user_auc * pos_len_list).sum() / pos_len_list.sum()
        return result


class AUC(LossMetric):
    r"""AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve
    .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
    Note:
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users.
    .. math::
        \mathrm {AUC} = \frac{\sum\limits_{i=1}^M rank_{i}
        - \frac {{M} \times {(M+1)}}{2}} {{{M} \times {N}}}
    :math:`M` is the number of positive samples.
    :math:`N` is the number of negative samples.
    :math:`rank_i` is the ascending rank of the ith positive sample.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('auc', dataobject)

    def metric_info(self, preds, trues):
        fps, tps = _binary_clf_curve(trues, preds)
        if len(fps) > 2:
            optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
            fps = fps[optimal_idxs]
            tps = tps[optimal_idxs]

        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        if fps[-1] <= 0:
            logger = getLogger()
            logger.warning("No negative samples in y_true, " "false positive value should be meaningless")
            fpr = np.repeat(np.nan, fps.shape)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            logger = getLogger()
            logger.warning("No positive samples in y_true, " "true positive value should be meaningless")
            tpr = np.repeat(np.nan, tps.shape)
        else:
            tpr = tps / tps[-1]

        result = sk_auc(fpr, tpr)
        return result

# Loss based Metrics #


class MAE(LossMetric):
    r"""`Mean absolute error regression loss`__
    .. __: https://en.wikipedia.org/wiki/Mean_absolute_error
    .. math::
        \mathrm{MAE}=\frac{1}{|{T}|} \sum_{(u, i) \in {T}}\left|\hat{r}_{u i}-r_{u i}\right|
    :math:`T` is the test set, :math:`\hat{r}_{u i}` is the score predicted by the model,
    and :math:`r_{u i}` the actual score of the test set.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('mae', dataobject)

    def metric_info(self, preds, trues):
        return mean_absolute_error(trues, preds)


class RMSE(LossMetric):
    r"""`Mean std error regression loss`__
   .. __: https://en.wikipedia.org/wiki/Root-mean-square_deviation
   .. math::
       \mathrm{RMSE} = \sqrt{\frac{1}{|{T}|} \sum_{(u, i) \in {T}}(\hat{r}_{u i}-r_{u i})^{2}}
   :math:`T` is the test set, :math:`\hat{r}_{u i}` is the score predicted by the model,
   and :math:`r_{u i}` the actual score of the test set.
   """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('rmse', dataobject)
    
    def metric_info(self, preds, trues):
        return np.sqrt(mean_squared_error(trues, preds))


class LogLoss(LossMetric):
    r"""`Log loss`__, aka logistic loss or cross-entropy loss
    .. __: http://wiki.fast.ai/index.php/Log_Loss
    .. math::
        -\log {P(y_t|y_p)} = -(({y_t}\ \log{y_p}) + {(1-y_t)}\ \log{(1 - y_p)})
    For a single sample, :math:`y_t` is true label in :math:`\{0,1\}`.
    :math:`y_p` is the estimated probability that :math:`y_t = 1`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('logloss', dataobject)

    def metric_info(self, preds, trues):
        eps = 1e-15
        preds = np.float64(preds)
        preds = np.clip(preds, eps, 1 - eps)
        loss = np.sum(-trues * np.log(preds) - (1 - trues) * np.log(1 - preds))
        return loss / len(preds)


class AveragePopularity:
    r"""It computes the average popularity of recommended items.

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>` and
                                            `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`_

    .. math::
        \mathrm{AveragePopularity}=\frac{1}{|U|} \sum_{u \in U_} \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}

    :math:`U` is total user set.

    :math:`R_{u}` is the recommended list of items for user u.

    :math:`\phi(i)` is the number of interaction of item i in training data.
    """

    def __init__(self, config):
        self.topk = config['topk']
        self.decimal_place = config['metric_decimal_place']

    def used_info(self, dataobject):
        """get the matrix of recommendation items and the popularity of items in training data"""
        item_counter = dataobject.get('data.count_items')
        item_matrix = dataobject.get('rec.items')
        return item_matrix.numpy(), dict(item_counter)

    def calculate_metric(self, dataobject):
        item_matrix, item_count = self.used_info(dataobject)
        result = self.metric_info(self.get_pop(item_matrix, item_count))
        metric_dict = self.topk_result('AveragePopularity', result)
        return metric_dict

    def get_pop(self, item_matrix, item_count):
        """convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.
        """
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = item_count.get(row[j], 0)
        return value

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, value):
        """match the metric value to the `k` and put them in `dictionary` form"""
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            metric_dict[key] = round(avg_result[k-1], self.decimal_place)
        return metric_dict


class ShannonEntropy:
    r"""This metric present the diversity of the recommendation items.
    It is the entropy over items' distribution.

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>` and
                                             `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`_

    .. math::
        \mathrm {ShannonEntropy}=-\sum_{i=1}^{n} p(i) \log p(i)

    :math:`p(i)` is the probability of recommending item i
    which is the number of item i in recommended list over all items.
    """

    def __init__(self, config):
        self.topk = config['topk']
        self.decimal_place = config['metric_decimal_place']

    def used_info(self, dataobject):
        """get the matrix of recommendation items."""
        item_matrix = dataobject.get('rec.items')
        return item_matrix.numpy()

    def calculate_metric(self, dataobject):
        item_matrix = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = '{}@{}'.format('ShannonEntropy', k)
            metric_dict[key] = round(self.get_entropy(item_matrix[:, :k]), self.decimal_place)
        return metric_dict

    def get_entropy(self, item_matrix):
        """convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.
        """
        item_count = dict(Counter(item_matrix.flatten()))
        total_num = item_matrix.shape[0]*item_matrix.shape[1]
        result = 0.0
        for cnt in item_count.values():
            p = cnt/total_num
            result += -p*np.log(p)
        return result/len(item_count)


metrics_dict = {
    'ndcg': NDCG,
    'hit': Hit,
    'precision': Precision,
    'map': MAP,
    'recall': Recall,
    'mrr': MRR,
    'rmse': RMSE,
    'mae': MAE,
    'logloss': LogLoss,
    'auc': AUC,
    'gauc': GAUC,
    'averagepopularity': AveragePopularity,
    'shannonentropy': ShannonEntropy
}
