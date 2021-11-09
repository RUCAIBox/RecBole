Evaluation Settings
===========================
Evaluation settings are designed to set parameters about model evaluation.



- ``eval_args (dict)``:  This parameter have 4 keys: ``group_by``, ``order``, ``split``, and ``mode``, 
  which respectively control the data grouping strategy, data ordering strategy, data splitting strategy
  and evaluation mode for model evaluation.   
  
  - ``group_by (str)``: decides how we group the data in `.inter`. Now we support two kinds of grouping strategies: ``['user', 'none']``. If the value of ``group_by`` is ``user``, the data will be grouped by the column of ``USER_ID_FIELD`` and split in user dimension. If the value is ``none``, the data won't be grouped. The default value is ``user``.   

  - ``order (str)``: decides how we sort the data in `.inter`. Now we support two kinds of ordering strategies: ``['RO', 'TO']``, which denotes the random ordering and temporal ordering. For ``RO``, we will shuffle the data and then split them in this order. For ``TO``, we will sort the data by the column of `TIME_FIELD` in ascending order and the split them in this order. The default value is ``RO``.
  
  - ``split (dict)``: decides how we split the data in `.inter`. Now we support two kinds of splitting strategies: ``['RS','LS']``, which denotes the ratio-based data splitting and leave-one-out data splitting. If the key of ``split`` is ``RS``, you need to set the splitting ratio like ``[0.8,0.1,0.1]``, ``[7,2,1]`` or ``[8,0,2]``, which denotes the ratio of training set, validation set and testing set respectively. If the key of ``split`` is ``LS``, now we support three kinds of ``LS`` mode: ``['valid_and_test', 'valid_only', 'test_only']`` and you should choose one mode as the value of ``LS``.  The default value of ``split`` is ``{'RS': [0.8,0.1,0.1]}``.
  
  - ``mode (str)``: decides the data range which we evaluate the model on. Now we support four kinds of evaluation mode: ``['full','unixxx','popxxx','labeled']``. ``full`` , ``unixxx`` and ``popxxx`` are designed for the evaluation on implicit feedback (data without label). For implicit feedback, we regard the items with observed interactions as positive items and those without observed interactions as negative items. ``full`` means evaluating the model on the set of all items. ``unixxx``, for example ``uni100``,  means uniformly sample 100 negative items for each positive item in testing set, and evaluate the model on these positive items with their sampled negative items. ``popxxx``, for example ``pop100``, means sample 100 negative items for each positive item in testing set based on item popularity (:obj:`Counter(item)` in `.inter` file), and evaluate the model on these positive items with their sampled negative items. Here the `xxx` must be an integer. For explicit feedback (data with label), you should set the mode as ``labeled`` and we will evaluate the model based on your label. The default value is ``full``.

- ``repeatable (bool)``: Whether to evaluate the result with a repeatable recommendation scene. Note that it is disabled for sequential models as the recommendation is already repeatable. For other models, defaults to ``False``.
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
- ``valid_metric (str)``: The evaluation metrics for early stopping. 
  It must be one of used ``metrics``. Defaults to ``'MRR@10'``.
- ``eval_batch_size (int)``: The evaluation batch size. Defaults to ``4096``.
- ``metric_decimal_place(int)``: The decimal place of metric score. Defaults to ``4``.

