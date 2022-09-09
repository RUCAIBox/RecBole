# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [DeepFM](https://recbole.io/docs/user_guide/model/context/deepfm.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-3,5e-3,1e-2]
    dropout_prob choice [0.0,0.1]
    mlp_hidden_size choice ['[128,128,128]','[256,256,256]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.1
    learning_rate: 0.005
    mlp_hidden_size: '[256,256,256]'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9015    logloss : 0.3115
    Test result:
    auc : 0.8997    logloss : 0.315

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.8994    logloss : 0.3124
    Test result:
    auc : 0.8982    logloss : 0.3152

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8978    logloss : 0.3205
    Test result:
    auc : 0.8975    logloss : 0.3222

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8962    logloss : 0.3222
    Test result:
    auc : 0.8949    logloss : 0.3252

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.8998    logloss : 0.3211
    Test result:
    auc : 0.8974    logloss : 0.3265

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.901    logloss : 0.3101
    Test result:
    auc : 0.8999    logloss : 0.3126

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9007    logloss : 0.3121
    Test result:
    auc : 0.8989    logloss : 0.3156

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8994    logloss : 0.313
    Test result:
    auc : 0.898    logloss : 0.3159

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8994    logloss : 0.3133
    Test result:
    auc : 0.8982    logloss : 0.316

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8995    logloss : 0.3124
    Test result:
    auc : 0.8979    logloss : 0.3155
  ```