# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [WideDeep](https://recbole.io/docs/user_guide/model/context/widedeep.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-4,1e-3,5e-3,1e-2]
    dropout_prob choice [0.0,0.2]
    mlp_hidden_size choice ['[64,64,64]','[128,128,128]','[256,256,256]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.2
    learning_rate: 0.005
    mlp_hidden_size: [256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.8959    logloss : 0.3339
    Test result:
    auc : 0.8942    logloss : 0.3377

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8983    logloss : 0.3132
    Test result:
    auc : 0.8966    logloss : 0.3167

    dropout_prob:0.2, learning_rate:0.01, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8988    logloss : 0.3129
    Test result:
    auc : 0.8963    logloss : 0.3171

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.8994    logloss : 0.3146
    Test result:
    auc : 0.8976    logloss : 0.3186

    dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8996    logloss : 0.313
    Test result:
    auc : 0.8982    logloss : 0.3159

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8965    logloss : 0.3205
    Test result:
    auc : 0.8965    logloss : 0.3224

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8981    logloss : 0.3189
    Test result:
    auc : 0.8969    logloss : 0.3225

    dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9008    logloss : 0.3116
    Test result:
    auc : 0.8987    logloss : 0.3155

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.8984    logloss : 0.318
    Test result:
    auc : 0.8967    logloss : 0.3211

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8949    logloss : 0.3294
    Test result:
    auc : 0.8938    logloss : 0.3325

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8987    logloss : 0.319
    Test result:
    auc : 0.8972    logloss : 0.3221

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.8977    logloss : 0.3188
    Test result:
    auc : 0.8954    logloss : 0.3236

    dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8988    logloss : 0.3202
    Test result:
    auc : 0.8974    logloss : 0.3231

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.899    logloss : 0.3183
    Test result:
    auc : 0.8972    logloss : 0.3223

    dropout_prob:0.2, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.9004    logloss : 0.3107
    Test result:
    auc : 0.8984    logloss : 0.3144

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8983    logloss : 0.3181
    Test result:
    auc : 0.8968    logloss : 0.3221

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8912    logloss : 0.3321
    Test result:
    auc : 0.8903    logloss : 0.3343

    dropout_prob:0.2, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9    logloss : 0.3129
    Test result:
    auc : 0.8979    logloss : 0.317
  ```