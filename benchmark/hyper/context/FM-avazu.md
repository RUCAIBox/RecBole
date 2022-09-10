# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [FM](https://recbole.io/docs/user_guide/model/context/fm.html)

- **Time cost**: 303.02s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,1e-4,2e-4,5e-4,1e-3,5e-3]
  ```

- **Best parameters**:

  ```
  learning_rate: 5e-05
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    learning_rate:0.0002
    Valid result:
    auc : 0.7932    logloss : 0.3672
    Test result:
    auc : 0.7912    logloss : 0.3673

    learning_rate:5e-05
    Valid result:
    auc : 0.7934    logloss : 0.3672
    Test result:
    auc : 0.7911    logloss : 0.3676

    learning_rate:0.0005
    Valid result:
    auc : 0.7934    logloss : 0.3661
    Test result:
    auc : 0.7918    logloss : 0.3663

    learning_rate:0.001
    Valid result:
    auc : 0.7922    logloss : 0.3653
    Test result:
    auc : 0.7901    logloss : 0.3655

    learning_rate:0.0001
    Valid result:
    auc : 0.7932    logloss : 0.3669
    Test result:
    auc : 0.791    logloss : 0.3671

    learning_rate:0.005
    Valid result:
    auc : 0.7927    logloss : 0.365
    Test result:
    auc : 0.7911    logloss : 0.3646
  ```

- **Logging Result**:

  ```yaml
    100%|██████████| 6/6 [28:47<00:00, 287.94s/trial, best loss: -0.7934]
    best params:  {'learning_rate': 5e-05}
    best result: 
    {'model': 'FM', 'best_valid_score': 0.7934, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7934), ('logloss', 0.3672)]), 'test_result': OrderedDict([('auc', 0.7911), ('logloss', 0.3676)])}
  ```
