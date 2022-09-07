# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [AFM](https://recbole.io/docs/user_guide/model/context/afm.html)

- **Time cost**: 3009.80s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,1e-4,5e-4]
    dropout_prob choice [0.0,0.1]
    attention_size choice [20,30]
    reg_weight choice [2,5]
  ```

- **Best parameters**:

  ```
    attention_size: 20
    dropout_prob: 0.1
    learning_rate: 0.0001
    reg_weight: 5
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    attention_size:20, dropout_prob:0.1, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.7891    logloss : 0.4554
    Test result:
    auc : 0.7911    logloss : 0.4544

    attention_size:30, dropout_prob:0.1, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.7891    logloss : 0.4554
    Test result:
    auc : 0.7912    logloss : 0.4544

    attention_size:30, dropout_prob:0.0, learning_rate:0.0001, reg_weight:2
    Valid result:
    auc : 0.789    logloss : 0.4556
    Test result:
    auc : 0.7911    logloss : 0.4546

    attention_size:20, dropout_prob:0.0, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.7889    logloss : 0.4556
    Test result:
    auc : 0.791    logloss : 0.4546

    attention_size:30, dropout_prob:0.1, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.789    logloss : 0.4555
    Test result:
    auc : 0.791    logloss : 0.4545

    attention_size:30, dropout_prob:0.0, learning_rate:5e-05, reg_weight:2
    Valid result:
    auc : 0.789    logloss : 0.4555
    Test result:
    auc : 0.791    logloss : 0.4545

    attention_size:30, dropout_prob:0.0, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.789    logloss : 0.4556
    Test result:
    auc : 0.7911    logloss : 0.4546

    attention_size:20, dropout_prob:0.1, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.7891    logloss : 0.4554
    Test result:
    auc : 0.7911    logloss : 0.4544

    attention_size:20, dropout_prob:0.0, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.7891    logloss : 0.4554
    Test result:
    auc : 0.7911    logloss : 0.4545

    attention_size:20, dropout_prob:0.0, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.789    logloss : 0.4555
    Test result:
    auc : 0.7911    logloss : 0.4545
  ```

- **Logging Result**:

  ```yaml
    42%|████▏     | 10/24 [8:54:31<12:28:20, 3207.18s/trial, best loss: -0.7891]
    best params:  {'attention_size': 20, 'dropout_prob': 0.1, 'learning_rate': 0.0001, 'reg_weight': 5}
    best result: 
    {'model': 'AFM', 'best_valid_score': 0.7891, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7891), ('logloss', 0.4554)]), 'test_result': OrderedDict([('auc', 0.7911), ('logloss', 0.4544)])}
  ```
