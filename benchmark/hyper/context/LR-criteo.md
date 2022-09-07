# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [LR](https://recbole.io/docs/user_guide/model/context/lr.html)

- **Time cost**: 856.77s/trial

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
  learning_rate:0.0005
  Valid result:
  auc : 0.7888    logloss : 0.4557
  Test result:
  auc : 0.7908    logloss : 0.4548

  learning_rate:0.0002
  Valid result:
  auc : 0.7888    logloss : 0.4557
  Test result:
  auc : 0.7908    logloss : 0.4548

  learning_rate:0.0001
  Valid result:
  auc : 0.7888    logloss : 0.4557
  Test result:
  auc : 0.7908    logloss : 0.4548

  learning_rate:0.005
  Valid result:
  auc : 0.786    logloss : 0.4586
  Test result:
  auc : 0.7882    logloss : 0.4575

  learning_rate:5e-05
  Valid result:
  auc : 0.7887    logloss : 0.4558
  Test result:
  auc : 0.7907    logloss : 0.4549

  learning_rate:0.001
  Valid result:
  auc : 0.7886    logloss : 0.4559
  Test result:
  auc : 0.7907    logloss : 0.4548
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 6/6 [1:35:42<00:00, 957.11s/trial, best loss: -0.7888]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'LR', 'best_valid_score': 0.7888, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7888), ('logloss', 0.4557)]), 'test_result': OrderedDict([('auc', 0.7908), ('logloss', 0.4548)])}
  ```
