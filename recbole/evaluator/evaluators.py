# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/04, 2020/08/11, 2020/12/9
# @Author  :   Kaiyuan Li, Yupeng Hou, Zhichao Feng
# @email   :   tsotfsk@outlook.com, houyupeng@ruc.edu.cn, fzcbupt@gmail.com


import torch
import numpy as np
from collections import ChainMap
from recbole.evaluator.metrics import metrics_dict
from recbole.evaluator.abstract_evaluator import GroupedEvalautor, IndividualEvaluator

# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP']}
# These metrics are typical in loss recommendations
loss_metrics = {metric.lower(): metric for metric in ['AUC', 'RMSE', 'MAE', 'LOGLOSS']}
# For GAUC
rank_metrics = {metric.lower(): metric for metric in ['GAUC']}

# group-based metrics
group_metrics = ChainMap(topk_metrics, rank_metrics)
# not group-based metrics
individual_metrics = ChainMap(loss_metrics)


class TopKEvaluator(GroupedEvalautor):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
       contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

   Note:
       The metrics used calculate group-based metrics which considers the metrics scores averaged
       across users. Some of them are also limited to k.

   """

    def __init__(self, config, metrics):
        super().__init__(config, metrics)

        self.topk = config['topk']
        self._check_args()

    def collect(self, interaction, scores_tensor):
        """collect the topk intermediate result of one batch, this function mainly
       implements padding and TopK finding. It is called at the end of each batch

       Args:
           interaction (Interaction): :class:`AbstractEvaluator` of the batch
           scores_tensor (tensor): the tensor of model output with size of `(N, )`

       """
        user_len_list = interaction.user_len_list
        scores_matrix = self.get_score_matrix(scores_tensor, user_len_list)

        # get topk
        _, topk_idx = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_idx

    def evaluate(self, batch_matrix_list, eval_data):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_len_list = eval_data.get_pos_len_list()
        topk_idx = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        assert len(pos_len_list) == len(topk_idx)
        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, topk_idx)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)

        return metric_dict

    def _check_args(self):

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError('topk must be a positive integer or a list of positive integers, '
                                     'but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (np.ndarray): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users

        Returns:
            np.ndarray: a matrix which contains the metrics result

        """
        pos_idx_matrix = (topk_index < pos_len_list.reshape(-1, 1))
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(pos_idx_matrix, pos_len_list)
            result_list.append(result)  # n_users x len(metrics) x len(ranks)
        result = np.stack(result_list, axis=0).mean(axis=1)  # len(metrics) x len(ranks)
        return result

    def __str__(self):
        msg = 'The TopK Evaluator Info:\n' + \
              '\tMetrics:[' + \
              ', '.join([topk_metrics[metric.lower()] for metric in self.metrics]) + \
              '], TopK:[' + \
              ', '.join(map(str, self.topk)) + \
              ']'
        return msg


class RankEvaluator(GroupedEvalautor):
    r"""Rank Evaluator is mainly used in ranking tasks except for topk tasks. Now, we support one
       rank metric containing `'GAUC'`.

       Note:
           The metrics used calculate group-based metrics which considers the metrics scores averaged
           across users except for top-k metrics.

       """

    def __init__(self, config, metrics):
        super().__init__(config, metrics)
        pass

    def get_user_pos_len_list(self, interaction, scores_tensor):
        """get number of positive items and all items in test set of each user

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`

        Returns:
            list: number of positive items,
            list: number of all items
        """
        pos_len_list = torch.Tensor(interaction.pos_len_list).to(scores_tensor.device)
        user_len_list = interaction.user_len_list
        return pos_len_list, user_len_list

    def get_pos_index(self, scores_tensor, pos_len_list, user_len_list):
        """get the index of positive items

       Args:
           scores_tensor (tensor): the tensor of model output with size of `(N, )`
           pos_len_list(list): number of positive items
           user_len_list(list): number of all items

       Returns:
           tensor: a matrix indicating whether the corresponding item is positive

       """
        scores_matrix = self.get_score_matrix(scores_tensor, user_len_list)
        _, n_index = torch.sort(scores_matrix, dim=-1, descending=True)
        pos_index = (n_index < pos_len_list.reshape(-1, 1))
        return pos_index

    def collect(self, interaction, scores_tensor):
        """collect the rank intermediate result of one batch, this function mainly implements ranking
        and calculating the sum of rank for positive items. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`

        """
        pos_len_list, user_len_list = self.get_user_pos_len_list(interaction, scores_tensor)
        pos_index = self.get_pos_index(scores_tensor, pos_len_list, user_len_list)
        index_list = torch.arange(1, pos_index.shape[1] + 1).to(pos_index.device)
        pos_rank_sum = torch.where(pos_index, index_list, torch.zeros_like(index_list)). \
            sum(axis=-1).reshape(-1, 1)
        return pos_rank_sum

    def evaluate(self, batch_matrix_list, eval_data):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data

        Returns:
            dict: such as ``{'GAUC:0.9286}``

        """
        pos_len_list = eval_data.get_pos_len_list()
        user_len_list = eval_data.get_user_len_list()
        pos_rank_sum = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(user_len_list, pos_len_list, pos_rank_sum)
        for metric, value in zip(self.metrics, result_list):
            key = '{}'.format(metric)
            metric_dict[key] = round(value, 4)

        return metric_dict

    def _calculate_metrics(self, user_len_list, pos_len_list, pos_rank_sum):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (np.ndarray): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users

        Returns:
            np.ndarray: a matrix which contains the metrics result

        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(user_len_list, pos_len_list, pos_rank_sum)
            result_list.append(result)
        return result_list

    def __str__(self):
        msg = 'The Rank Evaluator Info:\n' + \
              '\tMetrics:[' + \
              ', '.join([rank_metrics[metric.lower()] for metric in self.metrics]) + \
              '], TopK:[' + \
              ', '.join(map(str, self.topk)) + \
              ']'
        return msg


class LossEvaluator(IndividualEvaluator):
    r"""Loss Evaluator is mainly used in rating prediction and click through rate prediction. Now, we support four
       loss metrics which contain `'AUC', 'RMSE', 'MAE', 'LOGLOSS'`.

       Note:
           The metrics used do not calculate group-based metrics which considers the metrics scores averaged across users.
           They are also not limited to k. Instead, they calculate the scores on the entire prediction results regardless
           the users.

       """

    def __init__(self, config, metrics):
        super().__init__(config, metrics)

        self.label_field = config['LABEL_FIELD']

    def collect(self, interaction, pred_scores):
        """collect the loss intermediate result of one batch, this function mainly
        implements concatenating preds and trues. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            pred_scores (tensor): the tensor of model output with a size of `(N, )`

        Returns:
            tensor : a batch of scores with a size of `(N, 2)`

        """
        true_scores = interaction[self.label_field].to(pred_scores.device)
        assert len(true_scores) == len(pred_scores)
        return self.get_score_matrix(true_scores, pred_scores)

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

    def _calculate_metrics(self, trues, preds):
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

    def __str__(self):
        msg = 'The Loss Evaluator Info:\n' + \
              '\tMetrics:[' + \
              ', '.join([loss_metrics[metric.lower()] for metric in self.metrics]) + \
              ']'
        return msg


metric_eval_bind = [
    (topk_metrics, TopKEvaluator),
    (loss_metrics, LossEvaluator),
    (rank_metrics, RankEvaluator)
]
