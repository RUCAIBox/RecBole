import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .python import BaseLossEvaluator, BaseTopKEvaluator

# These metrics are typical in recommendations
topk_metrics = {metric.lower(): metric for metric in ['Hit', 'Recall', 'MRR', 'Precision', 'NDCG']}
loss_metrics = {metric.lower(): metric for metric in ['AUC', 'RMSE', 'MAE', 'LOGLOSS']}


class LossEvaluator(BaseLossEvaluator):

    def __init__(self, config, logger):
        super(LossEvaluator, self).__init__(config, logger)

    def evaluate(self, true_scores, pred_scores):
        """evalaute the loss metrics

        Args:
            true_scores (tensor): the true scores' list
            pred_scores (tensor): the predict scores' list

        Returns:
            dict: such as {'AUC': 0.83}
        """
        # get metrics
        metric_dict = {}
        result_list = self.metrics(true_scores, pred_scores)
        for metric, value in zip(self.metrics, result_list):
            key = '{}'.format(metric)
            metric_dict[key] = value
        return metric_dict


class TopKEvaluator(BaseTopKEvaluator):

    def __init__(self, config, logger):
        super(TopKEvaluator, self).__init__(config, logger)
        self.topk = config['topk']
        self.metrics = config['metrics']


    def evaluate(self, pos_len_list, score_tensor, user_idx_list):
        """ evalaute the topk metrics

        Args:
            pos_len_list (list): a list of the positive items' length
            score_tensor (tensor): a tensor of scores
            user_idx_list (list): a list of users' slice

        Returns:
            dict: such as { 'Hit@20': 0.3824, 'Recall@20': 0.0527
                            'Hit@10': 0.3153, 'Recall@10': 0.0329}
        """
        # intermediate variables
        score_list = []

        for slc in user_idx_list:
            score_list.append(score_tensor[slc])
        scores_matrix = pad_sequence(score_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k
        topk_index = topk_index.cpu().numpy()

        # get metrics
        metric_dict = {}
        result_list = self.eval_metrics(pos_len_list, topk_index)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = value[k - 1]
        return metric_dict

    def collect(self, metric_dict_list, user_num_list):
        """when using minibatch in training phase, you need to call this function to summarize the results

        Args:
            metric_dict_list (list): a list of metric dict
            user_num_list (list): a list of n_users

        Returns:
            dict: such as { 'Hit@5': 0.6666666666666666, 'MRR@5': 0.23796296296296293, 'Recall@5': 0.5277777777777778,
                            'Hit@3': 0.6666666666666666, 'MRR@3': 0.22685185185185186, 'Recall@3': 0.47222222222222215,
                            'Hit@1': 0.16666666666666666, 'MRR@1': 0.08333333333333333, 'Recall@1': 0.08333333333333333 }
        """
        tmp_result_list = []
        keys = list(metric_dict_list[0].keys())
        for result in metric_dict_list:
            tmp_result_list.append(list(result.values()))

        result_matrix = np.array(tmp_result_list)
        batch_size_matrix = np.array(user_num_list).reshape(-1, 1)
        assert result_matrix.shape[0] == batch_size_matrix.shape[0]

        # average
        weighted_matrix = result_matrix * batch_size_matrix
        metric_list = (np.sum(weighted_matrix, axis=0) / np.sum(batch_size_matrix)).tolist()

        # build metric_dict
        metric_dict = {}
        for method, score in zip(keys, metric_list):
            metric_dict[method] = score
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
                assert self.topk > 0, 'topk must be a pistive integer or a list of postive integers'
                self.topk = [self.topk]
            for topk in self.topk:
                assert topk > 0, 'topk must be a pistive integer or a list of postive integers'
        else:
            raise TypeError('The topk must be a integer, list or None')
