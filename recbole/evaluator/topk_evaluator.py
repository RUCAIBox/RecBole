# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/04, 2020/08/11
# @Author  :   Kaiyuan Li, Yupeng Hou
# @email   :   tsotfsk@outlook.com, houyupeng@ruc.edu.cn

"""
recbole.evaluator.topk_evaluator
################################
"""

import numpy as np
import torch
from recbole.evaluator.abstract_evaluator import AbstractEvaluator
from recbole.evaluator.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence

# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP']}


class TopKEvaluator(AbstractEvaluator):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which 
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.
    
    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged 
        across users. Some of them are also limited to k. 

    """
    def __init__(self, config):
        super().__init__(config)

        self.topk = config['topk']
        self._check_args()

    def collect(self, interaction, scores_tensor, full=False):
        """collect the topk intermediate result of one batch, this function mainly 
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`
            full (bool, optional): whether it is full sort. Default: False.

        """
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def evaluate(self, batch_matrix_list, eval_data):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_len_list = eval_data.get_pos_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        assert len(pos_len_list) == len(topk_index)
        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, topk_index)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)
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
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError('topk must be a positive integer or a list of positive integers, but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def metrics_info(self, pos_idx, pos_len):
        """get metrics result

        Args:
            pos_idx (np.ndarray): the bool index of all users' topk items that indicating the postive items are
                topk items or not
            pos_len (list): the length of all users' postivite items

        Returns:
            list: a list of matrix which record the results from `1` to `max(topk)`

        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(pos_idx, pos_len)
            result_list.append(result)
        return result_list

    def _calculate_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users

        Returns:
            np.ndarray: a matrix which contains the metrics result

        """

        pos_idx_matrix = (topk_index < pos_len_list.reshape(-1, 1))
        result_list = self.metrics_info(pos_idx_matrix, pos_len_list)  # n_users x len(metrics) x len(ranks)
        result = np.stack(result_list, axis=0).mean(axis=1)  # len(metrics) x len(ranks)
        return result

    def __str__(self):
        mesg = 'The TopK Evaluator Info:\n' + '\tMetrics:[' + ', '.join([topk_metrics[metric.lower()] for metric in self.metrics]) \
                + '], TopK:[' + ', '.join(map(str, self.topk)) +']'
        return mesg
