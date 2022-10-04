# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [xDeepFM](https://recbole.io/docs/user_guide/model/context/xdeepfm.html)

- **Time cost**: 410.64s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4,1e-3,5e-3,6e-3]
    dropout_prob choice [0.0,0.1]
    mlp_hidden_size choice ['[128,128,128]','[256,256,256]','[512,512,512]']
    cin_layer_size choice ['[60,60,60]','[100,100,100]']
    reg_weight choice [1e-5,5e-4]
  ```

- **Best parameters**:

  ```
    cin_layer_size: [60,60,60]
    dropout_prob: 0.1
    learning_rate: 0.005
    mlp_hidden_size: [512,512,512]
    reg_weight: 1e-05
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7971    logloss : 0.3616
    Test result:
    auc : 0.7948    logloss : 0.3619

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7909    logloss : 0.3656
    Test result:
    auc : 0.7876    logloss : 0.3665

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[256,256,256], reg_weight:0.0005
    Valid result:
    auc : 0.7929    logloss : 0.3643
    Test result:
    auc : 0.7903    logloss : 0.3646

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7964    logloss : 0.3619
    Test result:
    auc : 0.7945    logloss : 0.3619

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[256,256,256], reg_weight:0.0005
    Valid result:
    auc : 0.7933    logloss : 0.364
    Test result:
    auc : 0.791    logloss : 0.3642

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:0.0005
    Valid result:
    auc : 0.7901    logloss : 0.366
    Test result:
    auc : 0.7868    logloss : 0.3669

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7961    logloss : 0.3621
    Test result:
    auc : 0.7943    logloss : 0.3621

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.7915    logloss : 0.3651
    Test result:
    auc : 0.7883    logloss : 0.366

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:0.0005
    Valid result:
    auc : 0.7898    logloss : 0.3663
    Test result:
    auc : 0.7866    logloss : 0.3672

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7966    logloss : 0.3618
    Test result:
    auc : 0.7941    logloss : 0.3621
  ```

- **Logging Result**:

  ```yaml
    10%|█         | 10/96 [1:09:33<9:58:08, 417.31s/trial, best loss: -0.7971] 
    best params:  {'cin_layer_size': '[60,60,60]', 'dropout_prob': 0.1, 'learning_rate': 0.005, 'mlp_hidden_size': '[512,512,512]', 'reg_weight': 1e-05}
    best result: 
    {'model': 'xDeepFM', 'best_valid_score': 0.7971, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7971), ('logloss', 0.3616)]), 'test_result': OrderedDict([('auc', 0.7948), ('logloss', 0.3619)])}
  ```