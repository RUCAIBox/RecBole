Training & Evaluation Introduction
===================================

Training introduction
-----------------------
Multiple training strategies are supported by RecBole. For traditional CPU-based
collaborative filter models, non-gradient training is naturally applied. For
main-stream neural-based models, automatic gradient descent is well equipped
and set as default training strategy. Also two-stage training strategy is prepared
for pretraining-based models. In addition, users who need an unusual training strategy
can customize the ``Trainer`` and please refer to :doc:`../developer_guide/customize_trainers`
for more details.

Apart from flexible training strategies, an automatic hyper-parameter searching is
also supported. The implement of searching is fully based on `hyperopt <https://github.com/hyperopt/hyperopt>`_. 
Users can set the range of hyper-parameters in a config file with format of hyperopt
and the optimal hyper-parameter and result will be output. 
You can read :doc:`usage/parameter_tuning` for more information about hyper-parameter-tuning in RecBole.

To control the training method, we design a series of training parameters in config,
and you can read the :doc:`config/training_settings` for more information.


Evaluation introduction
-----------------------
The function of evaluation module is to implement commonly used evaluation
protocols for recommender systems. Since different models can be compared under
the same evaluation modules, RecBole standardizes the evaluation of recommender
systems.

Evaluation method
>>>>>>>>>>>>>>>>>>>>>>>

The evaluation method supported by RecBole is as following. Among them, the
first four rows correspond to the dataset splitting methods, while the last two
rows correspond to the ranking mechanism, namely a full ranking over all the
items or a sampled-based ranking.

==================       ========================================================
 Notation                   Explanation
==================       ========================================================
  RO                        Random Ordering
  TO                        Temporal Ordering
  LS                        Leave-one-out Splitting
  RS                        Ratio-based Splitting
  full                      full ranking with all item candidates
  uniN                      sample-based ranking: each positive item is paired with N sampled negative items in uniform distribution
  popN                      sample-based ranking: each positive item is paired with N sampled negative items in popularity distribution
==================       ========================================================

The parameters used to control the evaluation method are as follows:

- ``eval_args (dict)``: The overall evaluation settings. It contains all the setting of evaluation
  including ``split``, ``group_by``, ``order`` and ``mode``.

  - ``split (dict)``:  Control the splitting of dataset and the split ratio. The key is splitting method
    and value is the list of split ratio. The range of key is ``[RS,LS]``. Defaults to ``{'RS':[0.8, 0.1, 0.1]}``
  - ``group_by (str)``: Whether to split dataset with the group of user.
    Range in ``[None, user]`` and defaults to ``user``.
  - ``order (str)``: Control the ordering of data and affect the splitting of data.
    Range in ``['RO', 'TO']`` and defaults to ``RO``.
  - ``mode (str|dict)``: Control different candidates of ranking.
    Range in ``[labeled, full, unixxx, popxxx]`` and defaults to ``full``, which is equivalent to ``{'valid': 'full', 'test': 'full'}``.
 
- ``repeatable (bool)``: Whether to evaluate the result with a repeatable recommendation scene. Note that it is disabled for sequential models as the recommendation is already repeatable. For other models, defaults to ``False``.

Evaluation metrics
>>>>>>>>>>>>>>>>>>>>>>>>>>

RecBole supports both value-based and ranking-based evaluation metrics.

The value-based metrics (i.e., for rating prediction) include ``RMSE``, ``MAE``,
``AUC`` and ``LogLoss``, measuring the prediction difference between the true
and predicted values.

The ranking-based metrics (i.e., for top-k item recommendation) include the most
common ranking-aware metrics, such as ``Recall``, ``Precision``, ``Hit``,
``NDCG``, ``MAP``, ``MRR`` and ``GAUC``, measuring the ranking performance of the
generated recommendation lists by an algorithm. Besides, several ranking-based
non-accuracy metrics are supported to evaluate in different views, such as
``ItemCoverage``, ``AveragePopularity``, ``GiniIndex``, ``ShannonEntropy`` and ``TailPercentage``. 
More details about metrics can refer to :doc:`/recbole/recbole.evaluator.metrics`.

The parameters used to control the evaluation metrics are as follows:

- ``metrics (list or str)``: Evaluation metrics. Defaults to
  ``['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']``. Range in the following table:

  ==============    =================================================
  Type              Metrics
  ==============    =================================================
  Ranking-based     Recall, MRR, NDCG, Hit, MAP, Precision, GAUC, ItemCoverage, AveragePopularity, GiniIndex, ShannonEntropy, TailPercentage
  value-based       AUC, MAE, RMSE, LogLoss
  ==============    =================================================

  Note that value-based metrics and ranking-based metrics can not be used together.
- ``topk (list or int or None)``: The value of k for topk evaluation metrics.
  Defaults to ``10``.

For more details about evaluation settings, please read :doc:`config/evaluation_settings`
