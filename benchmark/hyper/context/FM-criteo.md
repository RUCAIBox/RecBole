# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [FM](https://recbole.io/docs/user_guide/model/context/fm.html)

- **Time cost**: 676.88s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,1e-4,2e-4,5e-4,1e-3,5e-3]
  ```

- **Best parameters**:

  ```
    learning_rate: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    learning_rate:0.0001
    Valid result:
    auc : 0.7893    logloss : 0.4556
    Test result:
    auc : 0.7914    logloss : 0.4547

    learning_rate:5e-05
    Valid result:
    auc : 0.7892    logloss : 0.4556
    Test result:
    auc : 0.7913    logloss : 0.4547

    learning_rate:0.001
    Valid result:
    auc : 0.7899    logloss : 0.4559
    Test result:
    auc : 0.792    logloss : 0.4553

    learning_rate:0.0002
    Valid result:
    auc : 0.7892    logloss : 0.4558
    Test result:
    auc : 0.7913    logloss : 0.4548

    learning_rate:0.005
    Valid result:
    auc : 0.77    logloss : 0.4903
    Test result:
    auc : 0.7715    logloss : 0.4895

    learning_rate:0.0005
    Valid result:
    auc : 0.7883    logloss : 0.4573
    Test result:
    auc : 0.7903    logloss : 0.4566
  ```

- **Logging Result**:

  ```yaml
    100%|██████████| 6/6 [1:05:09<00:00, 651.53s/trial, best loss: -0.7899]
    best params:  {'learning_rate': 0.001}
    best result: 
    {'model': 'FM', 'best_valid_score': 0.7899, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7899), ('logloss', 0.4559)]), 'test_result': OrderedDict([('auc', 0.792), ('logloss', 0.4553)])}
  ```
