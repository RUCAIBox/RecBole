# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [FNN](https://recbole.io/docs/user_guide/model/context/fnn.html)

- **Time cost**: 143.46s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-4,1e-3,3e-3,5e-3]
    dropout_prob choice [0.0,0.1]
    mlp_hidden_size choice ['[128,256,128]','[128,128,128]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.1
    learning_rate: 0.005
    mlp_hidden_size: [128,128,128]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7902    logloss : 0.3665
    Test result:
    auc : 0.7866    logloss : 0.3675

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7904    logloss : 0.3658
    Test result:
    auc : 0.788    logloss : 0.3661

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7925    logloss : 0.3648
    Test result:
    auc : 0.7903    logloss : 0.3649

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7903    logloss : 0.3658
    Test result:
    auc : 0.7881    logloss : 0.3659

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.788    logloss : 0.3672
    Test result:
    auc : 0.7849    logloss : 0.368

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7887    logloss : 0.3666
    Test result:
    auc : 0.7861    logloss : 0.3671

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7889    logloss : 0.3665
    Test result:
    auc : 0.7862    logloss : 0.367

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7883    logloss : 0.367
    Test result:
    auc : 0.7853    logloss : 0.3678

    dropout_prob:0.0, learning_rate:0.003, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7888    logloss : 0.367
    Test result:
    auc : 0.7865    logloss : 0.3673

    dropout_prob:0.1, learning_rate:0.003, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7886    logloss : 0.3673
    Test result:
    auc : 0.7863    logloss : 0.3675

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7888    logloss : 0.3667
    Test result:
    auc : 0.7858    logloss : 0.3674

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7917    logloss : 0.3654
    Test result:
    auc : 0.7896    logloss : 0.3653

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7886    logloss : 0.3669
    Test result:
    auc : 0.7855    logloss : 0.3677
  ```

- **Logging Result**:

  ```yaml
    81%|████████▏ | 13/16 [31:33<07:16, 145.66s/trial, best loss: -0.7925]
    best params:  {'dropout_prob': 0.0, 'learning_rate': 0.005, 'mlp_hidden_size': '[128,128,128]'}
    best result: 
    {'model': 'FNN', 'best_valid_score': 0.7925, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7925), ('logloss', 0.3648)]), 'test_result': OrderedDict([('auc', 0.7903), ('logloss', 0.3649)])}
  ```