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

from .metrics import metrics_dict
import numpy as np
import torch

# These metrics are typical in loss recommendations
loss_metrics = {metric.lower(): metric for metric in ['AUC', 'RMSE', 'MAE', 'LOGLOSS']}


class LossEvaluator(object):

    def __init__(self, config):
        self.metrics = config['metrics']
        self.label_field = config['LABEL_FIELD']

    def evaluate(self, interaction, pred_scores):
        """evalaute the loss metrics

        Args:
            true_scores (tensor): the true scores' list
            pred_scores (tensor): the predict scores' list

        Returns:
            dict: such as {'AUC': 0.83}
        """
        user_idx_list = interaction.user_idx_list
        pos_len_list = interaction.pos_len_list
        true_scores = []
        for pos_len, user_idx in zip(pos_len_list, user_idx_list):
            label = torch.tensor(pos_len * [1] + (user_idx.stop - user_idx.start - pos_len) * [0], dtype=torch.float)
            true_scores.append(label)
        true_scores = torch.cat(true_scores, 0).cuda()
        return torch.stack((true_scores, pred_scores.detach()), dim=1)

    def collect(self, batch_matrix_list, *args):

        concat = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        trues = concat[:, 0]
        preds = concat[:, 1]

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(trues, preds)
        for metric, value in zip(self.metrics, result_list):
            key = '{}'.format(metric)
            metric_dict[key] = value
        return metric_dict

    def _check_args(self):

        # Check eval_metric
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('eval_metric must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in loss_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

    def metrics_info(self, trues, preds):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(trues, preds)
            result_list.append(result)
        return result_list

    def _calculate_metrics(self, trues, preds):
        return self.metrics_info(trues, preds)