Customize Metrics
======================
Here we present how to develop a new metric, and apply it into RecBole.
Users can implement their customized metrics and combine the metrics with others for personal evaluation.

Here, it only takes three steps to incorporate a new metric and we introduce them step by step.

Create a New Metric Class
--------------------------
Then, we create a new class in the file :file:`~recbole.evaluator.metrics` and define the parameter in ``__init__()``:

.. code:: python

    from recbole.evaluator.base_metric import AbstractMetric
    class MyMetric(AbstractMetric):
        def __init__(self, config):

Set properties of the metric
-----------------------------
After that, we set the properties of metrics:

Set ``metric_need``
###################
It is a list that contains one or multiple string that corresponding to needed input of metrics.
For now, we support 9 inputs for metrics including both recommendation results and information of
dataset which are listed below.

==================       ========================================================
 Notation                   Explanation
==================       ========================================================
  rec.items                        K recommended items for each user
  rec.topk                         Boolean matrix indicating the existence of a recommended item in the test set
                                   and number of positive items for each user
  rec.meanrank                        Mean ranking of positive items for each user
  rec.score                        Pure output score
  data.num_items                      Number of item in dataset
  data.num_users                      Number of user in dataset
  data.count_items                    Interaction number of each item in training data
  data.count_users                    Interaction number of each user in training data
  data.label                          Pure label field of input data (Usually used with rec.score together)
==================       ========================================================

Set ``metric_type``
###################
It indicates whether the scores required by metric are grouped by user,
range in ``EvaluatorType.RANKING`` (for grouped ones) and ``EvaluatorType.VALUE`` (for non-grouped ones).
In current RecBole, all the "grouped" metrics are ranking-based and all the "non-grouped"
metrics are value-based. To keep with our paper, we adopted the more formal terms: ``RANKING`` and ``VALUE``.

Set ``smaller``
###############
It indicates whether the smaller metric value represents better performance, range in
``True`` and ``False``,  default to ``False``.

Example
#######
If we want to add a ranking-based metric named ``YourMetric`` which needs the recommended items and the
total item number, and the smaller ``YourMetric`` indicates better model performance, the code is shown below:

.. code:: python

    from recbole.evaluator.base_metric import AbstractMetric
    from recbole.utils import EvaluatorType
    class MyMetric(AbstractMetric):
        metric_type = EvaluatorType.RANKING
        metric_need = ['rec.items', 'data.num_items']
        smaller = True

        def __init__(self, config):

Implement calculate_metric(self, dataobject)
---------------------------------------------
All the computational process is defined in this function. The args is a packaged data object that
contains all the result above. We can treat it as a dict and get data from it by
``rec_items = dataobject.get('rec.items')`` . The returned value should be a dict with key of metric name
and value of final result. Note that the metric name should be lowercase.

Example code:

.. code:: python

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.

        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.

        Returns:
            dict: such as ``{'mymetric@10': 3153, 'mymetric@20': 0.3824}``
        """
        rec_items = dataobject.get('rec.items')
        # Add the logic of your metric here.

        return result_dict
