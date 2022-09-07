# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [DCN](https://recbole.io/docs/user_guide/model/context/dcn.html)

- **Time cost**: 203.67s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4,5e-4,1e-3,5e-3,6e-3]
    mlp_hidden_size choice ['[128,128,128]','[256,256,256]','[512,512,512]','[1024,1024,1024]']
    reg_weight choice [1,2,5]
    cross_layer_num choice [6]
    dropout_prob choice [0.1,0.2]
  ```

- **Best parameters**:

  ```
    cross_layer_num: 6
    dropout_prob: 0.1
    learning_rate: 0.005
    mlp_hidden_size: [512,512,512]
    reg_weight: 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[1024,1024,1024], reg_weight:2
    Valid result:
    auc : 0.7969    logloss : 0.362
    Test result:
    auc : 0.7943    logloss : 0.3626

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[256,256,256], reg_weight:5
    Valid result:
    auc : 0.7939    logloss : 0.3639
    Test result:
    auc : 0.7916    logloss : 0.3641

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[512,512,512], reg_weight:5
    Valid result:
    auc : 0.7861    logloss : 0.3701
    Test result:
    auc : 0.7828    logloss : 0.3708

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[512,512,512], reg_weight:2
    Valid result:
    auc : 0.7973    logloss : 0.3623
    Test result:
    auc : 0.7951    logloss : 0.3623

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[1024,1024,1024], reg_weight:2
    Valid result:
    auc : 0.7877    logloss : 0.3697
    Test result:
    auc : 0.7845    logloss : 0.3703

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[256,256,256], reg_weight:2
    Valid result:
    auc : 0.794    logloss : 0.3639
    Test result:
    auc : 0.7916    logloss : 0.3642

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.006, mlp_hidden_size:[1024,1024,1024], reg_weight:5
    Valid result:
    auc : 0.7928    logloss : 0.3648
    Test result:
    auc : 0.7901    logloss : 0.365

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.006, mlp_hidden_size:[1024,1024,1024], reg_weight:2
    Valid result:
    auc : 0.7929    logloss : 0.3647
    Test result:
    auc : 0.7901    logloss : 0.3651

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:5
    Valid result:
    auc : 0.7796    logloss : 0.3807
    Test result:
    auc : 0.7784    logloss : 0.3805

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[512,512,512], reg_weight:2
    Valid result:
    auc : 0.7961    logloss : 0.363
    Test result:
    auc : 0.7937    logloss : 0.3632

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[512,512,512], reg_weight:1
    Valid result:
    auc : 0.7962    logloss : 0.3628
    Test result:
    auc : 0.7939    logloss : 0.3629

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[1024,1024,1024], reg_weight:1
    Valid result:
    auc : 0.7847    logloss : 0.3726
    Test result:
    auc : 0.7814    logloss : 0.3729

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[1024,1024,1024], reg_weight:2
    Valid result:
    auc : 0.7944    logloss : 0.3639
    Test result:
    auc : 0.7921    logloss : 0.364

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[256,256,256], reg_weight:5
    Valid result:
    auc : 0.7895    logloss : 0.3679
    Test result:
    auc : 0.7865    logloss : 0.3684
  ```

- **Logging Result**:

  ```yaml
    12%|█▏        | 14/120 [48:47<6:09:24, 209.10s/trial, best loss: -0.7973]
    best params:  {'cross_layer_num': 6, 'dropout_prob': 0.1, 'learning_rate': 0.005, 'mlp_hidden_size': '[512,512,512]', 'reg_weight': 2}
    best result: 
    {'model': 'DCN', 'best_valid_score': 0.7973, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7973), ('logloss', 0.3623)]), 'test_result': OrderedDict([('auc', 0.7951), ('logloss', 0.3623)])}
  ```
