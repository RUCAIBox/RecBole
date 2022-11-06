# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [FiGNN](https://recbole.io/docs/user_guide/model/context/fignn.html)

- **Hyper-parameter searching** (hyper.test):

```yaml
  learning_rate choice [0.005,0.001,0.0005]
  attention_size choice [8,16,32]
  n_layers choice [2,3,4]
```

- **Best parameters**:

```
  attention_size: 16
  learning_rate: 0.005
  n_layers: 2
```

- **Hyper-parameter logging** (hyper.result):

```yaml
  attention_size:32, learning_rate:0.001, n_layers:2
  Valid result:
  auc : 0.7893    logloss : 0.3666
  Test result:
  auc : 0.7859    logloss : 0.3675

  attention_size:32, learning_rate:0.005, n_layers:3
  Valid result:
  auc : 0.7936    logloss : 0.3646
  Test result:
  auc : 0.7913    logloss : 0.3646

  attention_size:8, learning_rate:0.0005, n_layers:3
  Valid result:
  auc : 0.7843    logloss : 0.3696
  Test result:
  auc : 0.7808    logloss : 0.3705

  attention_size:16, learning_rate:0.001, n_layers:4
  Valid result:
  auc : 0.7892    logloss : 0.3667
  Test result:
  auc : 0.7859    logloss : 0.3675

  attention_size:8, learning_rate:0.005, n_layers:3
  Valid result:
  auc : 0.7929    logloss : 0.3642
  Test result:
  auc : 0.7904    logloss : 0.3645

  attention_size:32, learning_rate:0.005, n_layers:2
  Valid result:
  auc : 0.7938    logloss : 0.364
  Test result:
  auc : 0.7916    logloss : 0.3642

  attention_size:16, learning_rate:0.005, n_layers:3
  Valid result:
  auc : 0.7931    logloss : 0.3643
  Test result:
  auc : 0.7905    logloss : 0.3646

  attention_size:16, learning_rate:0.005, n_layers:2
  Valid result:
  auc : 0.7939    logloss : 0.364
  Test result:
  auc : 0.7913    logloss : 0.3642

  attention_size:16, learning_rate:0.001, n_layers:3
  Valid result:
  auc : 0.7892    logloss : 0.3665
  Test result:
  auc : 0.7859    logloss : 0.3673

  attention_size:8, learning_rate:0.001, n_layers:4
  Valid result:
  auc : 0.7883    logloss : 0.3673
  Test result:
  auc : 0.7848    logloss : 0.3682

  attention_size:8, learning_rate:0.005, n_layers:4
  Valid result:
  auc : 0.7916    logloss : 0.365
  Test result:
  auc : 0.7885    logloss : 0.3657

  attention_size:16, learning_rate:0.0005, n_layers:4
  Valid result:
  auc : 0.7854    logloss : 0.369
  Test result:
  auc : 0.782    logloss : 0.3698

  attention_size:8, learning_rate:0.001, n_layers:2
  Valid result:
  auc : 0.789    logloss : 0.3666
  Test result:
  auc : 0.7858    logloss : 0.3674

  attention_size:32, learning_rate:0.005, n_layers:4
  Valid result:
  auc : 0.7936    logloss : 0.3639
  Test result:
  auc : 0.7912    logloss : 0.364

  attention_size:8, learning_rate:0.0005, n_layers:4
  Valid result:
  auc : 0.7827    logloss : 0.3709
  Test result:
  auc : 0.779    logloss : 0.3718

  attention_size:8, learning_rate:0.001, n_layers:3
  Valid result:
  auc : 0.7885    logloss : 0.3669
  Test result:
  auc : 0.7854    logloss : 0.3676

  attention_size:32, learning_rate:0.0005, n_layers:3
  Valid result:
  auc : 0.7875    logloss : 0.3677
  Test result:
  auc : 0.7843    logloss : 0.3685

  attention_size:16, learning_rate:0.001, n_layers:2
  Valid result:
  auc : 0.7885    logloss : 0.367
  Test result:
  auc : 0.7852    logloss : 0.3678
```

- **Logging Result**:

```yaml
  67% 18/27 [1:47:34<53:47, 358.61s/trial, best loss: -0.7939]
  best params:  {'attention_size': 16, 'learning_rate': 0.005, 'n_layers': 2}
  best result: 
  {'model': 'FiGNN', 'best_valid_score': 0.7939, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7939), ('logloss', 0.364)]), 'test_result': OrderedDict([('auc', 0.7913), ('logloss', 0.3642)])}
```

