# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

# UPDATE
# @Time   : 2021/8/29
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

"""
recbole.evaluator.register
################################################
"""
import inspect
import sys


def cluster_info(module_name):
    """Collect information of all metrics, including:

        - ``metric_need``: Information needed to calculate this metric, the combination of ``rec.items, rec.topk,
          rec.meanrank, rec.score, data.num_items, data.num_users, data.count_items, data.count_users, data.label``.
        - ``metric_type``: Whether the scores required by metric are grouped by user, range in ``EvaluatorType.RANKING``
          and ``EvaluatorType.VALUE``.
        - ``smaller``: Whether the smaller metric value represents better performance,
          range in ``True`` and ``False``, default to ``False``.

    Note:
        For ``metric_type``: in current RecBole, all the "grouped-score" metrics are ranking-based and all the
        "non-grouped-score" metrics are value-based. To keep with our paper, we adopted the more formal terms:
        ``RANKING`` and ``VALUE``.

    Args:
        module_name (str): the name of module ``recbole.evaluator.metrics``.

    Returns:
        dict: Three dictionaries containing the above information
        and a dictionary matching metric names to metric classes.
    """
    smaller_m = []
    m_dict, m_info, m_types = {}, {}, {}
    metric_class = inspect.getmembers(
        sys.modules[module_name],
        lambda x: inspect.isclass(x) and x.__module__ == module_name,
    )
    for name, metric_cls in metric_class:
        name = name.lower()
        m_dict[name] = metric_cls
        if hasattr(metric_cls, "metric_need"):
            m_info[name] = metric_cls.metric_need
        else:
            raise AttributeError(f"Metric '{name}' has no attribute [metric_need].")
        if hasattr(metric_cls, "metric_type"):
            m_types[name] = metric_cls.metric_type
        else:
            raise AttributeError(f"Metric '{name}' has no attribute [metric_type].")
        if metric_cls.smaller is True:
            smaller_m.append(name)
    return smaller_m, m_info, m_types, m_dict


metric_module_name = "recbole.evaluator.metrics"
smaller_metrics, metric_information, metric_types, metrics_dict = cluster_info(
    metric_module_name
)


class Register(object):
    """Register module load the registry according to the metrics in config.
    It is a member of DataCollector.
    The DataCollector collect the resource that need for Evaluator under the guidance of Register
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self._build_register()

    def _build_register(self):
        for metric in self.metrics:
            metric_needs = metric_information[metric]
            for info in metric_needs:
                setattr(self, info, True)

    def has_metric(self, metric: str):
        if metric.lower() in self.metrics:
            return True
        else:
            return False

    def need(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        return False
