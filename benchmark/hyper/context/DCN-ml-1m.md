# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [DCN](https://recbole.io/docs/user_guide/model/context/dcn.html)

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
    dropout_prob: 0.2
    learning_rate: 0.001
    mlp_hidden_size: [512,512,512]
    reg_weight: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:1
    Valid result:
    auc : 0.9002    logloss : 0.3146
    Test result:
    auc : 0.8989    logloss : 0.3177

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:5
    Valid result:
    auc : 0.8971    logloss : 0.3236
    Test result:
    auc : 0.8967    logloss : 0.3249

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[512,512,512], reg_weight:1
    Valid result:
    auc : 0.8999    logloss : 0.3218
    Test result:
    auc : 0.8991    logloss : 0.3239

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:1
    Valid result:
    auc : 0.8996    logloss : 0.3152
    Test result:
    auc : 0.8986    logloss : 0.3176

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:1
    Valid result:
    auc : 0.8954    logloss : 0.3299
    Test result:
    auc : 0.895    logloss : 0.3307

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[512,512,512], reg_weight:1
    Valid result:
    auc : 0.9005    logloss : 0.3138
    Test result:
    auc : 0.8994    logloss : 0.3164

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:5
    Valid result:
    auc : 0.8954    logloss : 0.3299
    Test result:
    auc : 0.8951    logloss : 0.3306

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[256,256,256], reg_weight:2
    Valid result:
    auc : 0.8996    logloss : 0.3161
    Test result:
    auc : 0.8987    logloss : 0.3184

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:2
    Valid result:
    auc : 0.8999    logloss : 0.3212
    Test result:
    auc : 0.8983    logloss : 0.3248

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:2
    Valid result:
    auc : 0.8969    logloss : 0.3256
    Test result:
    auc : 0.8957    logloss : 0.3281

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[1024,1024,1024], reg_weight:5
    Valid result:
    auc : 0.8987    logloss : 0.3188
    Test result:
    auc : 0.8977    logloss : 0.3205

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1
    Valid result:
    auc : 0.8991    logloss : 0.3168
    Test result:
    auc : 0.898    logloss : 0.3195

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[1024,1024,1024], reg_weight:2
    Valid result:
    auc : 0.8965    logloss : 0.3345
    Test result:
    auc : 0.8956    logloss : 0.3362

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[1024,1024,1024], reg_weight:5
    Valid result:
    auc : 0.8982    logloss : 0.3237
    Test result:
    auc : 0.8977    logloss : 0.3253

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[1024,1024,1024], reg_weight:1
    Valid result:
    auc : 0.9004    logloss : 0.315
    Test result:
    auc : 0.8991    logloss : 0.3184

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[256,256,256], reg_weight:5
    Valid result:
    auc : 0.8991    logloss : 0.3195
    Test result:
    auc : 0.898    logloss : 0.3224
  ```