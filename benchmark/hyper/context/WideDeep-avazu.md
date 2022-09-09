# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [WideDeep](https://recbole.io/docs/user_guide/model/context/widedeep.html)

- **Time cost**: 184.47s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-4,1e-3,5e-3,1e-2]
    dropout_prob choice [0.0,0.2]
    mlp_hidden_size choice ['[64,64,64]','[128,128,128]','[256,256,256]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.0
    learning_rate: 0.005
    mlp_hidden_size: [256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7912    logloss : 0.3656
    Test result:
    auc : 0.7885    logloss : 0.3662

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7961    logloss : 0.3626
    Test result:
    auc : 0.7938    logloss : 0.3628

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7924    logloss : 0.3649
    Test result:
    auc : 0.7897    logloss : 0.3652

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7943    logloss : 0.3634
    Test result:
    auc : 0.7917    logloss : 0.3637

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7951    logloss : 0.3628
    Test result:
    auc : 0.7929    logloss : 0.363

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7956    logloss : 0.3625
    Test result:
    auc : 0.7931    logloss : 0.3629

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7902    logloss : 0.3664
    Test result:
    auc : 0.7869    logloss : 0.3674

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7926    logloss : 0.3647
    Test result:
    auc : 0.7895    logloss : 0.3654

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7949    logloss : 0.3629
    Test result:
    auc : 0.7932    logloss : 0.3627

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7943    logloss : 0.3633
    Test result:
    auc : 0.7914    logloss : 0.3638

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.787    logloss : 0.369
    Test result:
    auc : 0.7835    logloss : 0.37

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7894    logloss : 0.3665
    Test result:
    auc : 0.7859    logloss : 0.3675
  ```

- **Logging Result**:

  ```yaml
    50%|█████     | 12/24 [31:36<31:36, 158.06s/trial, best loss: -0.7961]
    best params:  {'dropout_prob': 0.0, 'learning_rate': 0.005, 'mlp_hidden_size': '[256,256,256]'}
    best result: 
    {'model': 'WideDeep', 'best_valid_score': 0.7961, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7961), ('logloss', 0.3626)]), 'test_result': OrderedDict([('auc', 0.7938), ('logloss', 0.3628)])}
  ```