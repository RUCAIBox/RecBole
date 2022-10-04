# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [AFM](https://recbole.io/docs/user_guide/model/context/afm.html)

- **Time cost**: 2191.35s/trial

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
    learning_rate: 5e-05
    reg_weight: 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    attention_size:30, dropout_prob:0.0, learning_rate:0.0005, reg_weight:5
    Valid result:
    auc : 0.7906    logloss : 0.3669
    Test result:
    auc : 0.788    logloss : 0.3675

    attention_size:20, dropout_prob:0.1, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.7917    logloss : 0.3653
    Test result:
    auc : 0.789    logloss : 0.3659

    attention_size:30, dropout_prob:0.0, learning_rate:0.0001, reg_weight:2
    Valid result:
    auc : 0.7912    logloss : 0.3665
    Test result:
    auc : 0.7886    logloss : 0.3671

    attention_size:20, dropout_prob:0.0, learning_rate:0.0005, reg_weight:5
    Valid result:
    auc : 0.7906    logloss : 0.3673
    Test result:
    auc : 0.7882    logloss : 0.3678

    attention_size:30, dropout_prob:0.0, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.7912    logloss : 0.3665
    Test result:
    auc : 0.7886    logloss : 0.3671

    attention_size:30, dropout_prob:0.1, learning_rate:0.0001, reg_weight:2
    Valid result:
    auc : 0.7917    logloss : 0.3654
    Test result:
    auc : 0.789    logloss : 0.366

    attention_size:20, dropout_prob:0.0, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.7906    logloss : 0.3673
    Test result:
    auc : 0.7882    logloss : 0.3678

    attention_size:30, dropout_prob:0.1, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.7917    logloss : 0.3655
    Test result:
    auc : 0.789    logloss : 0.3661

    attention_size:20, dropout_prob:0.1, learning_rate:5e-05, reg_weight:2
    Valid result:
    auc : 0.7919    logloss : 0.3654
    Test result:
    auc : 0.7892    logloss : 0.366

    attention_size:30, dropout_prob:0.0, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.7906    logloss : 0.3668
    Test result:
    auc : 0.7881    logloss : 0.3674

    attention_size:30, dropout_prob:0.1, learning_rate:0.0005, reg_weight:5
    Valid result:
    auc : 0.7916    logloss : 0.3654
    Test result:
    auc : 0.789    logloss : 0.366

    attention_size:20, dropout_prob:0.1, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.7918    logloss : 0.3654
    Test result:
    auc : 0.7892    logloss : 0.366

    attention_size:20, dropout_prob:0.1, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.7919    logloss : 0.3654
    Test result:
    auc : 0.7892    logloss : 0.366

    attention_size:20, dropout_prob:0.1, learning_rate:0.0005, reg_weight:5
    Valid result:
    auc : 0.7917    logloss : 0.3653
    Test result:
    auc : 0.789    logloss : 0.3659

    attention_size:20, dropout_prob:0.0, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.7913    logloss : 0.3669
    Test result:
    auc : 0.7888    logloss : 0.3675

    attention_size:30, dropout_prob:0.1, learning_rate:5e-05, reg_weight:2
    Valid result:
    auc : 0.7917    logloss : 0.3655
    Test result:
    auc : 0.789    logloss : 0.3661

    attention_size:20, dropout_prob:0.0, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.7912    logloss : 0.3665
    Test result:
    auc : 0.7888    logloss : 0.367

    attention_size:20, dropout_prob:0.0, learning_rate:0.0001, reg_weight:2
    Valid result:
    auc : 0.7912    logloss : 0.3665
    Test result:
    auc : 0.7888    logloss : 0.367

    attention_size:30, dropout_prob:0.1, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.7916    logloss : 0.3654
    Test result:
    auc : 0.789    logloss : 0.366
  ```

- **Logging Result**:

  ```yaml
    79%|███████▉  | 19/24 [8:58:00<2:21:34, 1698.95s/trial, best loss: -0.7919]
    best params:  {'attention_size': 20, 'dropout_prob': 0.1, 'learning_rate': 5e-05, 'reg_weight': 2}
    best result: 
    {'model': 'AFM', 'best_valid_score': 0.7919, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7919), ('logloss', 0.3654)]), 'test_result': OrderedDict([('auc', 0.7892), ('logloss', 0.366)])}
  ```
