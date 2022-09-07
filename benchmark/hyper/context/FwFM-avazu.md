# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/avazu.md)

- **Model**: [FwFM](https://recbole.io/docs/user_guide/model/context/fwfm.html)

- **Time cost**: 46115.61s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4,5e-4,1e-3,5e-3,1e-2]
    dropout_prob choice [0.0,0.2,0.4]
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.2
    learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.0, learning_rate:0.001
    Valid result:
    auc : 0.7915    logloss : 0.3673
    Test result:
    auc : 0.7886    logloss : 0.3679

    dropout_prob:0.4, learning_rate:0.01
    Valid result:
    auc : 0.7837    logloss : 0.371
    Test result:
    auc : 0.781    logloss : 0.3719

    dropout_prob:0.4, learning_rate:0.001
    Valid result:
    auc : 0.7916    logloss : 0.3672
    Test result:
    auc : 0.7887    logloss : 0.3678

    dropout_prob:0.2, learning_rate:0.0005
    Valid result:
    auc : 0.7918    logloss : 0.3671
    Test result:
    auc : 0.7889    logloss : 0.3677

    dropout_prob:0.4, learning_rate:0.0005
    Valid result:
    auc : 0.7918    logloss : 0.3671
    Test result:
    auc : 0.7889    logloss : 0.3677

    dropout_prob:0.2, learning_rate:0.0001
    Valid result:
    auc : 0.7917    logloss : 0.3676
    Test result:
    auc : 0.789    logloss : 0.3681

    dropout_prob:0.0, learning_rate:0.0001
    Valid result:
    auc : 0.7917    logloss : 0.3675
    Test result:
    auc : 0.789    logloss : 0.368

    dropout_prob:0.4, learning_rate:0.0001
    Valid result:
    auc : 0.7917    logloss : 0.3675
    Test result:
    auc : 0.789    logloss : 0.3681

    dropout_prob:0.2, learning_rate:0.01
    Valid result:
    auc : 0.7833    logloss : 0.3721
    Test result:
    auc : 0.7809    logloss : 0.3729

    dropout_prob:0.0, learning_rate:0.005
    Valid result:
    auc : 0.7882    logloss : 0.3684
    Test result:
    auc : 0.7854    logloss : 0.3692

    dropout_prob:0.4, learning_rate:0.005
    Valid result:
    auc : 0.7888    logloss : 0.3679
    Test result:
    auc : 0.7861    logloss : 0.3685

    dropout_prob:0.0, learning_rate:0.0005
    Valid result:
    auc : 0.7917    logloss : 0.3674
    Test result:
    auc : 0.7889    logloss : 0.3679

    dropout_prob:0.2, learning_rate:0.005
    Valid result:
    auc : 0.7884    logloss : 0.3683
    Test result:
    auc : 0.7855    logloss : 0.369

    dropout_prob:0.2, learning_rate:0.001
    Valid result:
    auc : 0.7915    logloss : 0.3671
    Test result:
    auc : 0.7887    logloss : 0.3677
  ```

- **Logging Result**:

  ```yaml
    93%|█████████▎| 14/15 [77:01:55<5:30:08, 19808.24s/trial, best loss: -0.7918]
    best params:  {'dropout_prob': 0.2, 'learning_rate': 0.0005}
    best result: 
    {'model': 'FwFM', 'best_valid_score': 0.7918, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7918), ('logloss', 0.3671)]), 'test_result': OrderedDict([('auc', 0.7889), ('logloss', 0.3677)])}
  ```