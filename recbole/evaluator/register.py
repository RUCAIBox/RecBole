# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/5
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

"""
recbole.evaluator.register
################################################
"""
from collections import ChainMap

metric_information = {
    'ndcg': ['rec.topk'],  # Sign in for topk ranking metrics
    'mrr': ['rec.topk'],
    'hit': ['rec.topk'],
    'recall': ['rec.topk'],
    'precision': ['rec.topk'],
    'map': ['rec.topk'],

    'itemcoverage': ['rec.items', 'data.num_items'],  # Sign in for topk non-accuracy metrics
    'averagepopularity': ['rec.items', 'data.count_items'],
    'giniindex': ['rec.items', 'data.num_items'],
    'shannonentropy': ['rec.items'],
    'tailpercentage': ['rec.items', 'data.count_items'],

    'gauc': ['rec.meanrank'],  # Sign in for full ranking metrics


    'auc': ['rec.score', 'data.label'],  # Sign in for scoring metrics
    'rmse': ['rec.score', 'data.label'],
    'mae': ['rec.score', 'data.label'],
    'logloss': ['rec.score', 'data.label']}
# These metrics are typical in top-k recommendations
topk_metrics = {metric.lower(): metric for metric in ['Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP',
                                                      'ItemCoverage', 'AveragePopularity', 'ShannonEntropy', 'GiniIndex']}
# These metrics are typical in loss recommendations
loss_metrics = {metric.lower(): metric for metric in ['AUC', 'RMSE', 'MAE', 'LOGLOSS']}
# For GAUC
rank_metrics = {metric.lower(): metric for metric in ['GAUC']}

# group-based metrics
group_metrics = ChainMap(topk_metrics, rank_metrics)
# not group-based metrics
individual_metrics = ChainMap(loss_metrics)


class Register(object):
    """ Register module load the registry according to the metrics in config.
        It is a member of DataCollector.
        The DataCollector collect the resource that need for Evaluator under the guidance of Register

        Note:
            If you want to implement a new metric, please sign the metric above like others !
        """
    def __init__(self, config):

        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self._build_register()

    def _build_register(self):
        for metric in self.metrics:
            if metric not in metric_information:
                raise ValueError("Metric {} not be signed up in /evaluator/register.py".format(metric))
            metric_needs = metric_information[metric]
            for metric_need in metric_needs:
                setattr(self, metric_need, True)

    def has_metric(self, metric: str):
        if metric.lower() in self.metrics:
            return True
        else:
            return False

    def need(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        return False

