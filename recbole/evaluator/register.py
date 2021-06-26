# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole.evaluator.register
################################################
"""

metric_information = {
    'ndcg': ['rec.topk'],  # Sign in for topk ranking metrics
    'mrr': ['rec.topk'],
    'hit': ['rec.topk'],
    'recall': ['rec.topk'],
    'precision': ['rec.topk'],
    'map': ['rec.topk'],


    'gauc': ['rec.meanrank'],  # Sign in for full ranking metrics


    'auc': ['rec.score', 'data.label'],  # Sign in for scoring metrics
    'rmse': ['rec.score', 'data.label'],
    'mae': ['rec.score', 'data.label'],
    'logloss': ['rec.score', 'data.label']}


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

