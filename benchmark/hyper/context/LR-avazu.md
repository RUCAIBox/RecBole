# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [LR](https://recbole.io/docs/user_guide/model/context/lr.html)

- **Time cost**: 326.80s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,1e-4,2e-4,5e-4,1e-3,5e-3]
  ```

- **Best parameters**:

  ```
    learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    learning_rate:0.001
    Valid result:
    auc : 0.7917    logloss : 0.3676
    Test result:
    auc : 0.7889    logloss : 0.3682

    learning_rate:0.0005
    Valid result:
    auc : 0.7918    logloss : 0.3671
    Test result:
    auc : 0.789    logloss : 0.3677

    learning_rate:0.0001
    Valid result:
    auc : 0.7917    logloss : 0.3675
    Test result:
    auc : 0.789    logloss : 0.3681

    learning_rate:0.0002
    Valid result:
    auc : 0.7918    logloss : 0.3674
    Test result:
    auc : 0.789    logloss : 0.368

    learning_rate:0.005
    Valid result:
    auc : 0.7911    logloss : 0.367
    Test result:
    auc : 0.7883    logloss : 0.3676

    learning_rate:5e-05
    Valid result:
    auc : 0.7917    logloss : 0.3676
    Test result:
    auc : 0.789    logloss : 0.3681
  ```

- **Logging Result**:

  ```yaml
    100%|██████████| 6/6 [1:49:07<00:00, 1091.32s/trial, best loss: -0.7918]
    best params:  {'learning_rate': 0.0005}
    best result: 
    {'model': 'LR', 'best_valid_score': 0.7918, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7918), ('logloss', 0.3671)]), 'test_result': OrderedDict([('auc', 0.789), ('logloss', 0.3677)])}
  ```
