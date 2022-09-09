# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [FFM](https://recbole.io/docs/user_guide/model/context/ffm.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4,5e-4,1e-3,5e-3,5e-2]
  ```

- **Best parameters**:

  ```
    learning_rate: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    learning_rate:0.05
    Valid result:
    auc : 0.8661    logloss : 0.3713
    Test result:
    auc : 0.8633    logloss : 0.3758

    learning_rate:0.001
    Valid result:
    auc : 0.9033    logloss : 0.3061
    Test result:
    auc : 0.9007    logloss : 0.3103

    learning_rate:0.005
    Valid result:
    auc : 0.9017    logloss : 0.3091
    Test result:
    auc : 0.898    logloss : 0.3145

    learning_rate:0.0001
    Valid result:
    auc : 0.8998    logloss : 0.3125
    Test result:
    auc : 0.8971    logloss : 0.3164

    learning_rate:0.0005
    Valid result:
    auc : 0.9026    logloss : 0.3076
    Test result:
    auc : 0.9002    logloss : 0.3115
  ```

- **Logging Result**:

  ```yaml
    100%|██████████| 5/5 [16:45<00:00, 201.00s/trial, best loss: -0.9033]
    best params:  {'learning_rate': 0.001}
    best result: 
    {'model': 'FFM', 'best_valid_score': 0.9033, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.9033), ('logloss', 0.3061)]), 'test_result': OrderedDict([('auc', 0.9007), ('logloss', 0.3103)])}
  ```