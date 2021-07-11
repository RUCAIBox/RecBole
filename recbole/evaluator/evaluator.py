# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.metrics import metrics_dict
from recbole.evaluator.collector import DataStruct
from recbole.evaluator.register import loss_metrics


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self._check_args()
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'GAUC': 0.9236}``

        """
        result_dict = {}
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        return result_dict

    def _check_args(self):
        # Check topk:
        if hasattr(self.config, 'topk'):
            topk = getattr(self.config, "topk")
            if isinstance(topk, (int, list)):
                if isinstance(self.topk, int):
                    self.topk = [self.topk]
                for topk in self.topk:
                    if topk <= 0:
                        raise ValueError(
                            'topk must be a positive integer or a list of positive integers, '
                            'but get `{}`'.format(topk)
                        )
            else:
                raise TypeError('The topk must be a integer, list')

        # Check Loss
        if set(self.metrics) & set(loss_metrics):
            is_full = 'full' in self.config['eval_setting']
            if is_full:
                raise NotImplementedError('Full sort evaluation do not match the metrics!')


