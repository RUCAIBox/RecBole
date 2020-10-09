# -*- encoding: utf-8 -*-
# @Time    :   2020/10/06
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

"""
recbox.evaluator.abstract_evaluator
################################
"""


class AbstractEvaluator(object):
    """Abstract Evaluator"""

    def __init__(self, config):
        self.metrics = config['metrics']

    def _check_args(self):
        """calculate the correct of the setting"""
        raise NotImplementedError

    def evaluate(self):
        """ evalaute the metrics"""
        raise NotImplementedError

    def collect(self):
        """calculate the metrics of all batches"""
        raise NotImplementedError

    def metrics_info(self):
        """get metrics result"""
        raise NotImplementedError

    def _calculate_metrics(self):
        """ to calculate the metrics"""
        raise NotImplementedError
