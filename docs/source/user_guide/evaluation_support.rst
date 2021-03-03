Evaluation Support
===========================

The function of evaluation module is to implement commonly used evaluation
protocols for recommender systems. Since different models can be compared under
the same evaluation modules, RecBole standardizes the evaluation of recommender
systems.


Evaluation Settings
-----------------------
The evaluation settings supported by RecBole is as following. Among them, the
first four rows correspond to the dataset splitting methods, while the last two
rows correspond to the ranking mechanism, namely a full ranking over all the
items or a sampled-based ranking.

==================       ========================================================
 Notation                   Explanation
==================       ========================================================
  RO_RS                     Random Ordering + Ratio-based Splitting
  TO_LS                     Temporal Ordering + Leave-one-out Splitting
  RO_LS                     Random Ordering + Leave-one-out Splitting
  TO_RS                     Temporal Ordering + Ratio-based Splitting
  full                      full ranking with all item candidates
  uniN                      sample-based ranking: each positive item is paired with N sampled negative items in uniform distribution
  popN                      sample-based ranking: each positive item is paired with N sampled negative items in popularity distribution
==================       ========================================================

The parameters used to control the evaluation settings are as follows:

- ``eval_setting (str)``: The evaluation settings. Defaults to ``'RO_RS,full'``.
  The parameter has two parts. The first part control the splitting methods,
  range in ``['RO_RS','TO_LS','RO_LS','TO_RS']``. The second part(optional)
  control the ranking mechanism, range in ``['full','uni100','uni1000','pop100','pop1000']``.
- ``group_by_user (bool)``: Whether the users are grouped.
  It must be ``True`` when ``eval_setting`` is in ``['RO_LS', 'TO_LS']``.
  Defaults to ``True``.
- ``spilt_ratio (list)``: The split ratio between train data, valid data and
  test data. It only take effects when the first part of ``eval_setting``
  is in ``['RO_RS', 'TO_RS']``. Defaults to ``[0.8, 0.1, 0.1]``.
- ``leave_one_num (int)``: It only take effects when the first part of
  ``eval_setting`` is in ``['RO_LS', 'TO_LS']``. Defaults to ``2``.

Evaluation Metrics
-----------------------

RecBole supports both value-based and ranking-based evaluation metrics.

The value-based metrics (i.e., for rating prediction) include ``RMSE``, ``MAE``,
``AUC`` and ``LogLoss``, measuring the prediction difference between the true
and predicted values.

The ranking-based metrics (i.e., for top-k item recommendation) include the most
common ranking-aware metrics, such as ``Recall``, ``Precision``, ``Hit``,
``NDCG``, ``MAP`` and ``MRR``, measuring the ranking performance of the
generated recommendation lists by an algorithm.

The parameters used to control the evaluation metrics are as follows:

- ``metrics (list or str)``: Evaluation metrics. Defaults to
  ``['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']``. Range in
  ``['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'AUC',
  'MAE', 'RMSE', 'LogLoss']``.
- ``topk (list or int or None)``: The value of k for topk evaluation metrics.
  Defaults to ``10``.
