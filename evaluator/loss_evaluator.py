from .metrics import metrics_dict
import numpy as np
import torch

# These metrics are typical in loss recommendations
loss_metrics = {metric.lower(): metric for metric in ['AUC', 'RMSE', 'MAE', 'LOGLOSS']}


class LossEvaluator(object):

    def __init__(self, config, logger):
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
        true_scores = interaction[self.label_field].cuda()
        return torch.stack((true_scores, pred_scores.detach()), dim=1)

    def collect(self, batch_matrix_list, *args):

        concat = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        trues = concat[:, 0]
        preds = concat[:, 1]

        # get metrics
        metric_dict = {}
        result_list = self.eval_metrics(trues, preds)
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

    def eval_metrics(self, trues, preds):
        return self.metrics_info(trues, preds)