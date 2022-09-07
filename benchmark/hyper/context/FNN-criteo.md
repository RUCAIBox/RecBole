# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [FNN](https://recbole.io/docs/user_guide/model/context/fnn.html)

- **Time cost**: 567.90s/trial

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
    dropout_prob:0.0, learning_rate:0.003, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7873    logloss : 0.458
    Test result:
    auc : 0.7892    logloss : 0.4571

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7873    logloss : 0.4574
    Test result:
    auc : 0.7894    logloss : 0.4566

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7871    logloss : 0.4597
    Test result:
    auc : 0.7894    logloss : 0.459

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7873    logloss : 0.4575
    Test result:
    auc : 0.7895    logloss : 0.4567

    dropout_prob:0.1, learning_rate:0.003, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7879    logloss : 0.4575
    Test result:
    auc : 0.7901    logloss : 0.4567

    dropout_prob:0.1, learning_rate:0.003, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7874    logloss : 0.459
    Test result:
    auc : 0.7897    logloss : 0.4582

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7888    logloss : 0.4558
    Test result:
    auc : 0.7909    logloss : 0.455

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7874    logloss : 0.4583
    Test result:
    auc : 0.7897    logloss : 0.4573

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7875    logloss : 0.4596
    Test result:
    auc : 0.7896    logloss : 0.4588

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7876    logloss : 0.4568
    Test result:
    auc : 0.7898    logloss : 0.4559

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7874    logloss : 0.457
    Test result:
    auc : 0.7897    logloss : 0.456

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7875    logloss : 0.4569
    Test result:
    auc : 0.7897    logloss : 0.456

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7872    logloss : 0.4575
    Test result:
    auc : 0.7893    logloss : 0.4567

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.7872    logloss : 0.4576
    Test result:
    auc : 0.7894    logloss : 0.4567

    dropout_prob:0.0, learning_rate:0.003, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7877    logloss : 0.4571
    Test result:
    auc : 0.79    logloss : 0.4561

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7875    logloss : 0.4569
    Test result:
    auc : 0.7898    logloss : 0.4559
  ```

- **Logging Result**:

  ```yaml
    100%|██████████| 16/16 [2:21:30<00:00, 530.68s/trial, best loss: -0.7888]
    best params:  {'dropout_prob': 0.1, 'learning_rate': 0.005, 'mlp_hidden_size': '[128,128,128]'}
    best result: 
    {'model': 'FNN', 'best_valid_score': 0.7888, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7888), ('logloss', 0.4558)]), 'test_result': OrderedDict([('auc', 0.7909), ('logloss', 0.455)])}
  ```