# -*- encoding: utf-8 -*-
# @Time    :   2020/12/9
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

# UPDATE
# @Time    :   2020/12/9
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.proxy_evaluator
#####################################
"""

from collections import ChainMap

from recbole.evaluator.evaluators import metric_eval_bind, group_metrics, individual_metrics


class ProxyEvaluator(object):
    r"""ProxyEvaluator is used to assign the corresponding evaluator according to the evaluation metrics,
    for example, TopkEvaluator for top-k metrics, and summarize the results of all evaluators.

   """

    def __init__(self, config):
        self.config = config
        self.valid_metrics = ChainMap(group_metrics, individual_metrics)
        self.metrics = self.config['metrics']
        self._check_args()
        self.evaluators = self.build()

    def build(self):
        """assign evaluators according to metrics.

        Returns:
            list: a list of evaluators.

        """
        evaluator_list = []
        metrics_list = [metric.lower() for metric in self.metrics]
        for metrics, evaluator in metric_eval_bind:
            used_metrics = [metric for metric in metrics_list if metric in metrics]
            if used_metrics:
                evaluator_list.append(evaluator(self.config, used_metrics))
        return evaluator_list

    def collect(self, interaction, scores):
        """collect the all used evaluators' intermediate result of one batch.

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores (tensor): the tensor of model output with size of `(N, )`

        """
        results = []
        for evaluator in self.evaluators:
            results.append(evaluator.collect(interaction, scores))
        return results

    def merge_batch_result(self, batch_matrix_list):
        """merge all the intermediate result got in `self.collect` for used evaluators separately.

        Args:
            batch_matrix_list (list): the results of all batches not separated

        Returns:
            dict: used evaluators' results of all batches

        """
        matrix_dict = {}
        for collect_list in batch_matrix_list:
            for i, value in enumerate(collect_list):
                matrix_dict.setdefault(i, []).append(value)

        return matrix_dict

    def evaluate(self, batch_matrix_list, eval_data):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'GAUC': 0.9236}``

        """
        matrix_dict = self.merge_batch_result(batch_matrix_list)
        result_dict = {}
        for i, evaluator in enumerate(self.evaluators):
            res = evaluator.evaluate(matrix_dict[i], eval_data)
            result_dict.update(res)
        return result_dict

    def _check_args(self):

        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in self.valid_metrics:
                raise ValueError("There is no metric named {}!".format(m))
