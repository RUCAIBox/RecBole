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
  attention_size: 32
  learning_rate: 0.005
  n_layers: 2
```

- **Hyper-parameter logging** (hyper.result):

```yaml
  attention_size:32, learning_rate:0.005, n_layers:2
  Valid result:
  auc : 0.7951    logloss : 0.4503
  Test result:
  auc : 0.7968    logloss : 0.4497

  attention_size:16, learning_rate:0.0005, n_layers:3
  Valid result:
  auc : 0.7887    logloss : 0.4558
  Test result:
  auc : 0.7909    logloss : 0.4547

  attention_size:16, learning_rate:0.001, n_layers:3
  Valid result:
  auc : 0.7916    logloss : 0.4533
  Test result:
  auc : 0.7937    logloss : 0.4524

  attention_size:8, learning_rate:0.005, n_layers:2
  Valid result:
  auc : 0.792    logloss : 0.4529
  Test result:
  auc : 0.794    logloss : 0.4521

  attention_size:32, learning_rate:0.001, n_layers:2
  Valid result:
  auc : 0.7918    logloss : 0.4531
  Test result:
  auc : 0.7937    logloss : 0.4524

  attention_size:16, learning_rate:0.005, n_layers:2
  Valid result:
  auc : 0.7942    logloss : 0.4509
  Test result:
  auc : 0.7958    logloss : 0.4504

  attention_size:32, learning_rate:0.001, n_layers:3
  Valid result:
  auc : 0.7912    logloss : 0.4535
  Test result:
  auc : 0.7932    logloss : 0.4527

  attention_size:8, learning_rate:0.001, n_layers:2
  Valid result:
  auc : 0.7904    logloss : 0.4543
  Test result:
  auc : 0.7927    logloss : 0.4532

  attention_size:16, learning_rate:0.0005, n_layers:4
  Valid result:
  auc : 0.7893    logloss : 0.4554
  Test result:
  auc : 0.7913    logloss : 0.4545

  attention_size:8, learning_rate:0.005, n_layers:4
  Valid result:
  auc : 0.7942    logloss : 0.4511
  Test result:
  auc : 0.7961    logloss : 0.4504
```

- **Logging Result**:

```yaml
  37% 10/27 [4:27:25<7:34:37, 1604.56s/trial, best loss: -0.7951]
  best params:  {'attention_size': 32, 'learning_rate': 0.005, 'n_layers': 2}
  best result: 
  {'model': 'FiGNN', 'best_valid_score': 0.7951, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7951), ('logloss', 0.4503)]), 'test_result': OrderedDict([('auc', 0.7968), ('logloss', 0.4497)])}
```

