# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [NFM](https://recbole.io/docs/user_guide/model/context/nfm.html)

- **Time cost**: 727.24s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,8e-5,1e-4,5e-4,1e-3]
    dropout_prob choice [0.1,0.2,0.3]
    mlp_hidden_size choice ['[20,20,20]','[40,40,40]','[50,50,50]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.1
    learning_rate: 0.001
    mlp_hidden_size: [40,40,40]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.1, learning_rate:8e-05, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7871    logloss : 0.4591
    Test result:
    auc : 0.7892    logloss : 0.4581

    dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.788    logloss : 0.4592
    Test result:
    auc : 0.7898    logloss : 0.4584

    dropout_prob:0.3, learning_rate:5e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7875    logloss : 0.459
    Test result:
    auc : 0.79    logloss : 0.4576

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7892    logloss : 0.456
    Test result:
    auc : 0.7913    logloss : 0.4551

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7881    logloss : 0.4575
    Test result:
    auc : 0.7902    logloss : 0.4566

    dropout_prob:0.2, learning_rate:5e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7871    logloss : 0.4592
    Test result:
    auc : 0.7893    logloss : 0.4581

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7887    logloss : 0.4564
    Test result:
    auc : 0.7906    logloss : 0.4556

    dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7878    logloss : 0.4585
    Test result:
    auc : 0.7898    logloss : 0.4575

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7889    logloss : 0.4563
    Test result:
    auc : 0.7911    logloss : 0.4554

    dropout_prob:0.3, learning_rate:0.001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.788    logloss : 0.4572
    Test result:
    auc : 0.7899    logloss : 0.4564

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7885    logloss : 0.4565
    Test result:
    auc : 0.7905    logloss : 0.4555

    dropout_prob:0.1, learning_rate:8e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7872    logloss : 0.459
    Test result:
    auc : 0.7892    logloss : 0.4581

    dropout_prob:0.3, learning_rate:0.0005, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.787    logloss : 0.4586
    Test result:
    auc : 0.7891    logloss : 0.4577

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7879    logloss : 0.4576
    Test result:
    auc : 0.7899    logloss : 0.4566
  ```

- **Logging Result**:

  ```yaml
    31%|███       | 14/45 [2:21:13<5:12:43, 605.28s/trial, best loss: -0.7892]
    best params:  {'dropout_prob': 0.1, 'learning_rate': 0.001, 'mlp_hidden_size': '[40,40,40]'}
    best result: 
    {'model': 'NFM', 'best_valid_score': 0.7892, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7892), ('logloss', 0.456)]), 'test_result': OrderedDict([('auc', 0.7913), ('logloss', 0.4551)])}
  ```