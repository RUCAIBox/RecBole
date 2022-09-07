# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [PNN](https://recbole.io/docs/user_guide/model/context/pnn.html)

- **Time cost**: 181.65s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-3,3e-3,5e-3,6e-3,1e-2]
    dropout_prob choice [0.0,0.1]
    mlp_hidden_size choice ['[64,64,64]','[128,128,128]','[256,256,256]']
    reg_weight choice [0.0]
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.1
    learning_rate' 0.005
    mlp_hidden_size: [128,128,128]
    reg_weight: 0.0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.7988    logloss : 0.3614
    Test result:
    auc : 0.7965    logloss : 0.3617

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.7941    logloss : 0.3634
    Test result:
    auc : 0.7909    logloss : 0.3641

    dropout_prob:0.0, learning_rate:0.003, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.7971    logloss : 0.3615
    Test result:
    auc : 0.7946    logloss : 0.3619

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.7975    logloss : 0.3614
    Test result:
    auc : 0.7955    logloss : 0.3617

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.7976    logloss : 0.3613
    Test result:
    auc : 0.7951    logloss : 0.3617

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.7937    logloss : 0.3638
    Test result:
    auc : 0.7905    logloss : 0.3645

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.7976    logloss : 0.3615
    Test result:
    auc : 0.7952    logloss : 0.3618

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.798    logloss : 0.3612
    Test result:
    auc : 0.7958    logloss : 0.3614

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.7948    logloss : 0.363
    Test result:
    auc : 0.792    logloss : 0.3635

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.7986    logloss : 0.3606
    Test result:
    auc : 0.797    logloss : 0.3606
  ```

- **Logging Result**:

  ```yaml
    33%|███▎      | 10/30 [31:02<1:02:05, 186.29s/trial, best loss: -0.7988]
    best params:  {'dropout_prob': 0.1, 'learning_rate': 0.005, 'mlp_hidden_size': '[128,128,128]', 'reg_weight': 0.0}
    best result: 
    {'model': 'PNN', 'best_valid_score': 0.7988, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7988), ('logloss', 0.3614)]), 'test_result': OrderedDict([('auc', 0.7965), ('logloss', 0.3617)])}
  ```