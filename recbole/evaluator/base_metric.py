# @Time   : 2020/10/21
# @Author : Kaiyuan Li
# @email  : tsotfsk@outlook.com

# UPDATE
# @Time   : 2020/10/21, 2021/8/29
# @Author : Kaiyuan Li, Zhichao Feng
# @email  : tsotfsk@outlook.com, fzcbupt@gmail.com

"""
recbole.evaluator.abstract_metric
#####################################
"""

import torch
from recbole.utils import EvaluatorType


class AbstractMetric(object):
    """:class:`AbstractMetric` is the base object of all metrics. If you want to
        implement a metric, you should inherit this class.

    Args:
        config (Config): the config of evaluator.
    """

    smaller = False

    def __init__(self, config):
        self.decimal_place = config["metric_decimal_place"]

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.

        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.

        Returns:
            dict: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
        """
        raise NotImplementedError("Method [calculate_metric] should be implemented.")


class TopkMetric(AbstractMetric):
    """:class:`TopkMetric` is a base object of top-k metrics. If you want to
    implement an top-k metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.topk"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the bool matrix indicating whether the corresponding item is positive
        and number of positive items for each user.
        """
        rec_mat = dataobject.get("rec.topk")
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        return topk_idx.to(torch.bool).numpy(), pos_len_list.squeeze(-1).numpy()

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = "{}@{}".format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict

    def metric_info(self, pos_index, pos_len=None):
        """Calculate the value of the metric.

        Args:
            pos_index(numpy.ndarray): a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th \
            highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
            pos_len(numpy.ndarray): a vector representing the number of positive items per user, shape of ``(n_users,)``.

        Returns:
            numpy.ndarray: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
        """
        raise NotImplementedError(
            "Method [metric_info] of top-k metric should be implemented."
        )


class LossMetric(AbstractMetric):
    """:class:`LossMetric` is a base object of loss based metrics and AUC. If you want to
    implement an loss based metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """

    metric_type = EvaluatorType.VALUE
    metric_need = ["rec.score", "data.label"]

    def __init__(self, config):
        super().__init__(config)

    def used_info(self, dataobject):
        """Get scores that model predicted and the ground truth."""
        preds = dataobject.get("rec.score")
        trues = dataobject.get("data.label")

        return preds.squeeze(-1).numpy(), trues.squeeze(-1).numpy()

    def output_metric(self, metric, dataobject):
        preds, trues = self.used_info(dataobject)
        result = self.metric_info(preds, trues)
        return {metric: round(result, self.decimal_place)}

    def metric_info(self, preds, trues):
        """Calculate the value of the metric.

        Args:
            preds (numpy.ndarray): the scores predicted by model, a one-dimensional vector.
            trues (numpy.ndarray): the label of items, which has the same shape as ``preds``.

        Returns:
            float: The value of the metric.
        """
        raise NotImplementedError(
            "Method [metric_info] of loss-based metric should be implemented."
        )
