from ..metrics import metrics_dict
from ..utils import TOPK_ARGS, ArrayIndex
import numpy as np


class BaseTopKEvaluator(object):

    def __init__(self, config, logger):
        self.metrics = config['metrics']
        self.topk = config['topk']

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
            # args = getattr(TOPK_ARGS, metric.upper())
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(pos_idx, pos_len)
            result_list.append(result)
        return result_list

    def eval_metrics(self, pos_len_list, topk_index):
        """ to evaluate the metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users

        Returns:
            np.ndarray: a matrix which contains the metrics result
        """

        pos_idx_matrix = np.zeros_like(topk_index)
        for idx, (pos_len, topk) in enumerate(zip(pos_len_list, topk_index)):
            _, _, pos_idx = np.intersect1d(np.arange(pos_len), topk, return_indices=True)
            pos_idx_matrix[idx, pos_idx] = 1
        result_list = self.metrics_info(pos_idx_matrix.astype(bool), np.array(pos_len_list))  # n_users x len(metrics) x len(ranks)
        result = np.stack(result_list, axis=0).mean(axis=1)  # len(metrics) x len(ranks)
        return result


class BaseLossEvaluator(object):

    def __init__(self, config, logger):
        self.metrics = config['eval_metric']

    def metrics_info(self, trues, preds):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(trues, preds)
            result_list.append(result)
        return result_list

    def eval_metrics(self, trues, preds):
        return self.metrics_info(trues, preds)