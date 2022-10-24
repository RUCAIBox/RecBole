# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

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
  dropout_prob: 0.1
  learning_rate: 0.001
```

- **Hyper-parameter logging** (hyper.result):

```yaml
  attention_size:16, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.8708    logloss : 0.3503
  Test result:
  auc : 0.8748    logloss : 0.3453

  attention_size:32, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.8928    logloss : 0.3332
  Test result:
  auc : 0.8971    logloss : 0.3247

  attention_size:8, dropout_prob:0.1, learning_rate:5e-05
  Valid result:
  auc : 0.886    logloss : 0.363
  Test result:
  auc : 0.8891    logloss : 0.349

  attention_size:8, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.8897    logloss : 0.3379
  Test result:
  auc : 0.8938    logloss : 0.329

  attention_size:8, dropout_prob:0.1, learning_rate:0.001
  Valid result:
  auc : 0.8897    logloss : 0.3379
  Test result:
  auc : 0.8938    logloss : 0.329

  attention_size:32, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.8867    logloss : 0.3443
  Test result:
  auc : 0.8905    logloss : 0.3347

  attention_size:16, dropout_prob:0.1, learning_rate:5e-05
  Valid result:
  auc : 0.8708    logloss : 0.3503
  Test result:
  auc : 0.8748    logloss : 0.3453

  attention_size:16, dropout_prob:0.1, learning_rate:0.001
  Valid result:
  auc : 0.8939    logloss : 0.3394
  Test result:
  auc : 0.8976    logloss : 0.3325

  attention_size:8, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.886    logloss : 0.363
  Test result:
  auc : 0.8891    logloss : 0.349

  attention_size:16, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.8939    logloss : 0.3394
  Test result:
  auc : 0.8976    logloss : 0.3325

  attention_size:32, dropout_prob:0.1, learning_rate:0.001
  Valid result:
  auc : 0.8928    logloss : 0.3332
  Test result:
  auc : 0.8971    logloss : 0.3247

  attention_size:32, dropout_prob:0.1, learning_rate:5e-05
  Valid result:
  auc : 0.8867    logloss : 0.3443
  Test result:
  auc : 0.8905    logloss : 0.3347
```

- **Logging Result**:

```yaml
  100%|█████████████████████████████████████████████████████████████████████████████████████████| 12/12 [1:12:27<00:00, 362.25s/trial, best loss: -0.8939]
  best params:  {'attention_size': 16, 'dropout_prob': 0.1, 'learning_rate': 0.001}
  best result: 
  {'model': 'FiGNN', 'best_valid_score': 0.8939, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.8939), ('logloss', 0.3394)]), 'test_result': OrderedDict([('auc', 0.8976), ('logloss', 0.3325)])}
```

