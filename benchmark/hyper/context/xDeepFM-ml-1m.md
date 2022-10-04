# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [xDeepFM](https://recbole.io/docs/user_guide/model/context/xdeepfm.html)

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
    cin_layer_size: [100,100,100]
    dropout_prob: 0.1
    learning_rate: 0.001
    mlp_hidden_size: [512,512,512]
    reg_weight: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.898    logloss : 0.3186
    Test result:
    auc : 0.8967    logloss : 0.3214

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.8961    logloss : 0.3307
    Test result:
    auc : 0.8948    logloss : 0.3335

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.8961    logloss : 0.3193
    Test result:
    auc : 0.8941    logloss : 0.3227

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.8946    logloss : 0.3214
    Test result:
    auc : 0.8923    logloss : 0.3249

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.8926    logloss : 0.3208
    Test result:
    auc : 0.8884    logloss : 0.3267

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.8984    logloss : 0.3178
    Test result:
    auc : 0.896    logloss : 0.3224

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.8882    logloss : 0.3262
    Test result:
    auc : 0.8843    logloss : 0.3317

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.8915    logloss : 0.3314
    Test result:
    auc : 0.8897    logloss : 0.3349

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[512,512,512], reg_weight:0.0005
    Valid result:
    auc : 0.8983    logloss : 0.3206
    Test result:
    auc : 0.8973    logloss : 0.3231

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[512,512,512], reg_weight:0.0005
    Valid result:
    auc : 0.8992    logloss : 0.3141
    Test result:
    auc : 0.8986    logloss : 0.3157

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.8966    logloss : 0.32
    Test result:
    auc : 0.8945    logloss : 0.3239

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.8808    logloss : 0.344
    Test result:
    auc : 0.8783    logloss : 0.3486

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.8943    logloss : 0.3194
    Test result:
    auc : 0.8907    logloss : 0.3244

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.8973    logloss : 0.3184
    Test result:
    auc : 0.8954    logloss : 0.3219

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.8968    logloss : 0.3211
    Test result:
    auc : 0.8938    logloss : 0.3264

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.8861    logloss : 0.3402
    Test result:
    auc : 0.8852    logloss : 0.341

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.8967    logloss : 0.3213
    Test result:
    auc : 0.8947    logloss : 0.3252

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256], reg_weight:0.0005
    Valid result:
    auc : 0.8962    logloss : 0.318
    Test result:
    auc : 0.8951    logloss : 0.3204

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.8967    logloss : 0.3192
    Test result:
    auc : 0.8947    logloss : 0.3225

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[128,128,128], reg_weight:0.0005
    Valid result:
    auc : 0.8864    logloss : 0.3327
    Test result:
    auc : 0.8826    logloss : 0.3378
  ```