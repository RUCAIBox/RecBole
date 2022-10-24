# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

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
  dropout_prob: 0.1
  learning_rate: 0.001
```

- **Hyper-parameter logging** (hyper.result):

```yaml
  attention_size:32, dropout_prob:0.1, learning_rate:0.001
  Valid result:
  auc : 0.7935    logloss : 0.3642
  Test result:
  auc : 0.7908    logloss : 0.3647

  attention_size:8, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.7928    logloss : 0.3647
  Test result:
  auc : 0.7899    logloss : 0.3652

  attention_size:16, dropout_prob:0.1, learning_rate:0.001
  Valid result:
  auc : 0.7928    logloss : 0.3646
  Test result:
  auc : 0.7901    logloss : 0.3651

  attention_size:16, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.7928    logloss : 0.3646
  Test result:
  auc : 0.7901    logloss : 0.3651

  attention_size:16, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.7837    logloss : 0.37
  Test result:
  auc : 0.7804    logloss : 0.3707

  attention_size:32, dropout_prob:0.1, learning_rate:5e-05
  Valid result:
  auc : 0.7784    logloss : 0.3819
  Test result:
  auc : 0.7761    logloss : 0.383

  attention_size:8, dropout_prob:0.1, learning_rate:5e-05
  Valid result:
  auc : 0.7825    logloss : 0.3706
  Test result:
  auc : 0.7789    logloss : 0.3714

  attention_size:8, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.7825    logloss : 0.3706
  Test result:
  auc : 0.7789    logloss : 0.3714

  attention_size:32, dropout_prob:0.0, learning_rate:0.001
  Valid result:
  auc : 0.7935    logloss : 0.3642
  Test result:
  auc : 0.7908    logloss : 0.3647

  attention_size:32, dropout_prob:0.0, learning_rate:5e-05
  Valid result:
  auc : 0.7784    logloss : 0.3819
  Test result:
  auc : 0.7761    logloss : 0.383
```

- **Logging Result**:

```yaml
  83% 10/12 [1:27:32<17:30, 525.30s/trial, best loss: -0.7935]
  best params:  {'attention_size': 32, 'dropout_prob': 0.1, 'learning_rate': 0.001}
  best result: 
  {'model': 'FiGNN', 'best_valid_score': 0.7935, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7935), ('logloss', 0.3642)]), 'test_result': OrderedDict([('auc', 0.7908), ('logloss', 0.3647)])}
```

