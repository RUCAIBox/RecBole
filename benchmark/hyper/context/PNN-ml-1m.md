# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [PNN](https://recbole.io/docs/user_guide/model/context/pnn.html)

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
    learning_rate: 0.005
    mlp_hidden_size: [256,256,256]
    reg_weight: 0.0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.8985    logloss : 0.3132
    Test result:
    auc : 0.8956    logloss : 0.3178

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.8973    logloss : 0.3157
    Test result:
    auc : 0.8951    logloss : 0.3197

    dropout_prob:0.1, learning_rate:0.003, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.897    logloss : 0.3153
    Test result:
    auc : 0.8946    logloss : 0.3191

    dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.8983    logloss : 0.3132
    Test result:
    auc : 0.8952    logloss : 0.3176

    dropout_prob:0.1, learning_rate:0.003, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.8984    logloss : 0.3148
    Test result:
    auc : 0.8961    logloss : 0.3183

    dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8985    logloss : 0.3172
    Test result:
    auc : 0.8962    logloss : 0.3219

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8952    logloss : 0.3191
    Test result:
    auc : 0.8934    logloss : 0.3221

    dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.8987    logloss : 0.3135
    Test result:
    auc : 0.8956    logloss : 0.3186

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.8968    logloss : 0.3177
    Test result:
    auc : 0.8937    logloss : 0.3231

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8987    logloss : 0.315
    Test result:
    auc : 0.8964    logloss : 0.3191

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.8977    logloss : 0.3152
    Test result:
    auc : 0.8956    logloss : 0.3188

    dropout_prob:0.1, learning_rate:0.003, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8988    logloss : 0.3137
    Test result:
    auc : 0.8955    logloss : 0.3192

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8975    logloss : 0.3138
    Test result:
    auc : 0.8945    logloss : 0.3178

    dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.8982    logloss : 0.3127
    Test result:
    auc : 0.8952    logloss : 0.3174

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.897    logloss : 0.3147
    Test result:
    auc : 0.8949    logloss : 0.3185

    dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8973    logloss : 0.3142
    Test result:
    auc : 0.8954    logloss : 0.318

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.8991    logloss : 0.3147
    Test result:
    auc : 0.8967    logloss : 0.3192

    dropout_prob:0.0, learning_rate:0.003, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.8967    logloss : 0.3198
    Test result:
    auc : 0.894    logloss : 0.3245

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.8979    logloss : 0.3136
    Test result:
    auc : 0.8957    logloss : 0.3169

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8944    logloss : 0.3269
    Test result:
    auc : 0.8924    logloss : 0.3304

    dropout_prob:0.0, learning_rate:0.003, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8979    logloss : 0.3182
    Test result:
    auc : 0.8961    logloss : 0.322

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.8948    logloss : 0.3286
    Test result:
    auc : 0.8926    logloss : 0.3326

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[64,64,64], reg_weight:0.0
    Valid result:
    auc : 0.8938    logloss : 0.3266
    Test result:
    auc : 0.8926    logloss : 0.3285

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.8973    logloss : 0.3142
    Test result:
    auc : 0.8945    logloss : 0.3184

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.8956    logloss : 0.3217
    Test result:
    auc : 0.893    logloss : 0.326

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[256,256,256], reg_weight:0.0
    Valid result:
    auc : 0.8966    logloss : 0.3159
    Test result:
    auc : 0.8946    logloss : 0.3193

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[128,128,128], reg_weight:0.0
    Valid result:
    auc : 0.8968    logloss : 0.3165
    Test result:
    auc : 0.8951    logloss : 0.3198
  ```