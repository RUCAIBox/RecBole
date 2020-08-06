# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/04           2020/08/06
# @Author  :   Kaiyuan Li           Yupeng Hou
# @email   :   tsotfsk@outlook.com  houyupeng@ruc.edu.cn

import numpy as np
import torch
from .metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence

# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Hit', 'Recall', 'MRR', 'Precision', 'NDCG']}


class TopKEvaluator(object):

    def __init__(self, config):
        self.topk = config['topk']
        self.metrics = config['metrics']

    def evaluate(self, intercation, score_tensor):
        """ evalaute the topk metrics

        Args:
            score_tensor (tensor): a tensor of scores
            pos_len_list (list): a list of the positive items' length
            user_len_list (list): a list of users' length

        Returns:
            dict: such as { 'Hit@20': 0.3824, 'Recall@20': 0.0527
                            'Hit@10': 0.3153, 'Recall@10': 0.0329}
        """
        # intermediate variables
        user_len_list = intercation.user_len_list
        pos_len_list = intercation.pos_len_list

        all_score_list = torch.split(score_tensor, user_len_list, dim=0)

        score_list = []
        for i, score in enumerate(all_score_list):
            if pos_len_list[i] > 0:
                score_list.append(score)
        scores_matrix = pad_sequence(score_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def collect(self, batch_matrix_list, batch_pos_list):

        pos_len_list = np.concatenate(batch_pos_list)
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        assert len(pos_len_list) == len(topk_index)
        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, topk_index)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = value[k - 1]
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
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                assert self.topk > 0, 'topk must be a positive integer or a list of positive integers'
                self.topk = [self.topk]
            for topk in self.topk:
                assert topk > 0, 'topk must be a positive integer or a list of positive integers'
        else:
            raise TypeError('The topk must be a integer, list or None')

    def metrics_info(self, pos_idx, pos_len):
        """get one users's metrics result

        Args:
            topk_idx (np.ndarray): the int index of topk items
            pos_idx (np.ndarray): the bool index of postivite items
            pos_len (int): the length of postivite items

        Returns:
            list: a list of metrics result
        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(pos_idx, pos_len)
            result_list.append(result)
        return result_list

    def _calculate_metrics(self, pos_len_list, topk_index):
        """ to evaluate the metrics by users

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