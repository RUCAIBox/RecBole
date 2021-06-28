# -*- encoding: utf-8 -*-
# @Time    :   2020/10/21
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/10/21, 2021/6/25
# @Author  :   Kaiyuan Li, Zhichao Feng
# @email   :   tsotfsk@outlook.com, fzcbupt@gmail.com

"""
recbole.evaluator.abstract_metric
#####################################
"""

import torch


class TopkMetric(object):
    """:class:`TopkMetric` is a base object of top-k metrics. If you want to
    implement an top-k metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.

    """
    def __init__(self, config):
        self.topk = config['topk']
        self.indice = [max(self.topk), 1, 1]
        self.decimal_place = config['metric_decimal_place']

    def used_info(self, dataobject):
        """get the bool matrix indicating whether the corresponding item is positive"""
        rec_mat = dataobject.get('rec.topk')
        topk_idx, shapes, pos_len_list = torch.split(rec_mat, self.indice, dim=1)
        pos_idx_matrix = (topk_idx >= (shapes - pos_len_list).reshape(-1, 1))

        return pos_idx_matrix.numpy(), pos_len_list.squeeze().numpy()

    def topk_result(self, metric, value):
        """match the metric value to the `k` and put them in `dictionary` form"""
        metric_dict = {}
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            metric_dict[key] = round(value[k - 1], self.decimal_place)
        return metric_dict


class LossMetric(object):
    """:class:`LossMetric` is a base object of loss based metrics and AUC. If you want to
    implement an loss based metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.

    """
    def __init__(self, config):
        self.decimal_place = config['metric_decimal_place']

    def used_info(self, dataobject):
        """get scores that model predicted and the ground truth"""
        preds = dataobject.get('rec.score')
        trues = dataobject.get('data.label')

        return preds.squeeze().numpy(), trues.squeeze().numpy()
