# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/04   2020/08/09
# @Author  :   Kaiyuan Li   Zhichao Feng
# @email   :   tsotfsk@outlook.com  fzcbupt@gmail.com

"""
recbole.evaluator.loss_evaluator
################################
"""

import numpy as np
import torch
from recbole.evaluator.abstract_evaluator import AbstractEvaluator
from recbole.evaluator.metrics import metrics_dict

# These metrics are typical in loss recommendations
loss_metrics = {metric.lower(): metric for metric in ['AUC', 'RMSE', 'MAE', 'LOGLOSS']}


class LossEvaluator(AbstractEvaluator):
    r"""Loss Evaluator is mainly used in rating prediction and click through rate prediction. Now, we support four 
    loss metrics which contain `'AUC', 'RMSE', 'MAE', 'LOGLOSS'`.

    Note:
        The metrics used do not calculate group-based metrics which considers the metrics scores averaged across users.
        They are also not limited to k. Instead, they calculate the scores on the entire prediction results regardless the users.

    """
    def __init__(self, config):
        super().__init__(config)
        
        self.label_field = config['LABEL_FIELD']
        self._check_args()

    def collect(self, interaction, pred_scores):
        """collect the loss intermediate result of one batch, this function mainly 
        implements concatenating preds and trues. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            pred_scores (tensor): the tensor of model output with a size of `(N, )`

        Returns:
            tensor : a batch of socres with a size of `(N, 2)`

        """
        true_scores = interaction[self.label_field].to(pred_scores.device)
        assert len(true_scores) == len(pred_scores)
        return torch.stack((true_scores, pred_scores.detach()), dim=1)

    def evaluate(self, batch_matrix_list, *args):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches

        Returns:
            dict: such as {'AUC': 0.83}

        """
        concat = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        trues = concat[:, 0]
        preds = concat[:, 1]

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(trues, preds)
        for metric, value in zip(self.metrics, result_list):
            key = '{}'.format(metric)
            metric_dict[key] = round(value, 4)
        return metric_dict

    def _check_args(self):

        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in loss_metrics:
                raise ValueError("There is no loss metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

    def metrics_info(self, trues, preds):
        """get metrics result

        Args:
            trues (np.ndarray): the true scores' list
            preds (np.ndarray): the predict scores' list

        Returns:
            list: a list of metrics result

        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(trues, preds)
            result_list.append(result)
        return result_list

    def _calculate_metrics(self, trues, preds):
        return self.metrics_info(trues, preds)

    def __str__(self):
        mesg = 'The Loss Evaluator Info:\n' + '\tMetrics:[' + ', '.join([loss_metrics[metric.lower()] for metric in self.metrics]) + ']'
        return mesg
