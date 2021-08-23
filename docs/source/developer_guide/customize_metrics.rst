Customize Metrics
======================
Here we present how to develop a new metric, and apply it into RecBole.
Users can implement their customized metrics and combine the metrics with others for personal evaluation.

Here, it only takes three steps to incorporate a new metric and we introduce them step by step.


Sign in Your Metric in Register
--------------------------------
To begin with, we must add a new line in :obj:`~recbole.evaluator.register.metric_information`:
All the metrics are registered by :obj:`metric_information` which is a dict. Keys are the name of
metrics and should be lowercase. Value is a list which contain one or multiple string that corresponding
to needed input of metrics.

For now, we support 9 inputs for metrics including both recommendation results and information of dataset
which are listed below.

==================       ========================================================
 Notation                   Explanation
==================       ========================================================
  rec.items                        K recommended items for each user
  rec.topk                        K recommended items and number of positive items for each user
  rec.meanrank                        Mean ranking of positive items for each user
  rec.score                        Pure output score
  data.num_items                      Number of item in dataset
  data.num_users                      Number of user in dataset
  data.count_items                    Interaction number of each item in training data
  data.count_users                    Interaction number of each user in training data
  data.label                          Pure label field of input data (Usually used with rec.score together)
==================       ========================================================

For example, if we want to add a metric named ``YourMetric`` which need the recommended items
and the total item number, we can sign in the metric as follow.

.. code:: python

    metric_information = {
    'ndcg': ['rec.topk'],  # Sign in for topk ranking metrics
    'mrr': ['rec.topk'],
    'map': ['rec.topk'],

    'itemcoverage': ['rec.items', 'data.num_items'],  # Sign in for topk non-accuracy metrics

    'yourmetric': ['rec.items', 'data.num_items'] # Sign in your customized metric
    }


Create a New Metric Class
--------------------------
Then, we create a new class in the file :file:`~recbole.evaluator.metrics` and define the parameter in
``__init__()``

.. code:: python

    from recbole.evaluator.base_metric import AbstractMetric
    class MyMetric(AbstractMetric):
        def __init__(self, config):


Implement calculate_metric(self, dataobject)
---------------------------------------------
All the computational process is defined in this function. The args is a packaged data object that
contains all the result above. We can treat it as a dict and get data from it by
``rec_items = dataobject.get('rec.items')`` . The returned value should be a dict with key of metric name
and value of final result.

Example code:

.. code:: python

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.

        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.

        Returns:
            dict: such as ``{'Mymetric@10': 3153, 'MyMetric@20': 0.3824}``
        """
        rec_items = dataobject.get('rec.items')
        # Add the logic of your metric here.

        return result_dict
