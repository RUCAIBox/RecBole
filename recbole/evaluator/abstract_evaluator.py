# -*- encoding: utf-8 -*-
# @Time    :   2020/10/21
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/10/21, 2020/12/18
# @Author  :   Kaiyuan Li, Zhichao Feng
# @email   :   tsotfsk@outlook.com, fzcbupt@gmail.com

"""
recbole.evaluator.abstract_evaluator
#####################################
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class BaseEvaluator(object):
    """:class:`BaseEvaluator` is an object which supports
    the evaluation of the model. It is called by :class:`Trainer`.

    Note:
        If you want to inherit this class and implement your own evaluator class,
        you must implement the following functions.

    Args:
        config (Config): The config of evaluator.

    """

    def __init__(self, config, metrics):
        self.metrics = metrics
        self.full = ('full' in config['eval_setting'])
        self.precision = config['metric_decimal_place']

    def collect(self, *args):
        """get the intermediate results for each batch, it is called at the end of each batch"""
        raise NotImplementedError

    def evaluate(self, *args):
        """calculate the metrics of all batches, it is called at the end of each epoch"""
        raise NotImplementedError

    def _calculate_metrics(self, *args):
        """ to calculate the metrics"""
        raise NotImplementedError


class GroupedEvaluator(BaseEvaluator):
    """:class:`GroupedEvaluator` is an object which supports the evaluation of the model.

    Note:
        If you want to implement a new group-based metric,
        you may need to inherit this class

    """

    def __init__(self, config, metrics):
        super().__init__(config, metrics)
        pass

    def sample_collect(self, scores_tensor, user_len_list):
        """padding scores_tensor. It is called when evaluation sample distribution is `uniform` or `popularity`.

        """
        scores_list = torch.split(scores_tensor, user_len_list, dim=0)
        padding_score = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # n_users x items
        return padding_score

    def full_sort_collect(self, scores_tensor, user_len_list):
        """it is called when evaluation sample distribution is `full`.

        """
        return scores_tensor.view(len(user_len_list), -1)

    def get_score_matrix(self, scores_tensor, user_len_list):
        """get score matrix.

        Args:
           scores_tensor (tensor): the tensor of model output with size of `(N, )`
           user_len_list(list): number of all items

       """
        if self.full:
            scores_matrix = self.full_sort_collect(scores_tensor, user_len_list)
        else:
            scores_matrix = self.sample_collect(scores_tensor, user_len_list)
        return scores_matrix


class IndividualEvaluator(BaseEvaluator):
    """:class:`IndividualEvaluator` is an object which supports the evaluation of the model.

    Note:
        If you want to implement a new non-group-based metric,
        you may need to inherit this class

    """

    def __init__(self, config, metrics):
        super().__init__(config, metrics)
        self._check_args()

    def sample_collect(self, true_scores, pred_scores):
        """It is called when evaluation sample distribution is `uniform` or `popularity`.

        """
        return torch.stack((true_scores, pred_scores.detach()), dim=1)

    def full_sort_collect(self, true_scores, pred_scores):
        """it is called when evaluation sample distribution is `full`.

        """
        raise NotImplementedError('full sort can\'t use IndividualEvaluator')

    def get_score_matrix(self, true_scores, pred_scores):
        """get score matrix

        Args:
           true_scores (tensor): the label of predicted items
           pred_scores (tensor): the tensor of model output with a size of `(N, )`

       """
        if self.full:
            scores_matrix = self.full_sort_collect(true_scores, pred_scores)
        else:
            scores_matrix = self.sample_collect(true_scores, pred_scores)

        return scores_matrix

    def _check_args(self):
        if self.full:
            raise NotImplementedError('full sort can\'t use IndividualEvaluator')
