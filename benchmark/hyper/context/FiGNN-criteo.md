# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [FiGNN](https://recbole.io/docs/user_guide/model/context/fignn.html)

- **Hyper-parameter searching** (hyper.test):

```yaml
  learning_rate choice [5e-5,1e-3]
  dropout_prob choice [0.0,0.1]
  attention_size choice [8,16,32]
```

- **Best parameters**:

```
  attention_size: 16
  dropout_prob: 0.0
  learning_rate: 0.001
```

- **Hyper-parameter logging** (hyper.result):

```yaml
  attention_size:8, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.7934    logloss : 0.4516
  Test result:
  auc : 0.7953    logloss : 0.4509

  attention_size:32, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.7864    logloss : 0.458
  Test result:
  auc : 0.7884    logloss : 0.457

  attention_size:16, dropout_prob:0.1, learning_rate:5e-05
  Valid result:
  auc : 0.7852    logloss : 0.4597
  Test result:
  auc : 0.7872    logloss : 0.4587

  attention_size:8, dropout_prob:0.1, learning_rate:0.001
  Valid result:
  auc : 0.7934    logloss : 0.4516
  Test result:
  auc : 0.7953    logloss : 0.4509

  attention_size:32, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.7947    logloss : 0.4513
  Test result:
  auc : 0.7965    logloss : 0.4506

  attention_size:32, dropout_prob:0.1, learning_rate:5e-05
  Valid result:
  auc : 0.7864    logloss : 0.458
  Test result:
  auc : 0.7884    logloss : 0.457

  attention_size:16, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.7852    logloss : 0.4597
  Test result:
  auc : 0.7872    logloss : 0.4587

  attention_size:8, dropout_prob:0.1, learning_rate:5e-05
  Valid result:
  auc : 0.786    logloss : 0.4587
  Test result:
  auc : 0.788    logloss : 0.4577

  attention_size:16, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.7951    logloss : 0.4509
  Test result:
  auc : 0.7967    logloss : 0.4503

  attention_size:16, dropout_prob:0.1, learning_rate:0.001
  Valid result:
  auc : 0.7951    logloss : 0.4509
  Test result:
  auc : 0.7967    logloss : 0.4503

  attention_size:8, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.786    logloss : 0.4587
  Test result:
  auc : 0.788    logloss : 0.4577

  attention_size:32, dropout_prob:0.1, learning_rate:0.001
  Valid result:
  auc : 0.7947    logloss : 0.4513
  Test result:
  auc : 0.7965    logloss : 0.4506
```

- **Logging Result**:

```yaml
  100%|████████████████████████████████████████████████████████████████████████████████████████| 12/12 [3:29:27<00:00, 1047.32s/trial, best loss: -0.7951]
  best params:  {'attention_size': 16, 'dropout_prob': 0.0, 'learning_rate': 0.001}
  best result: 
  {'model': 'FiGNN', 'best_valid_score': 0.7951, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7951), ('logloss', 0.4509)]), 'test_result': OrderedDict([('auc', 0.7967), ('logloss', 0.4503)])}
```

