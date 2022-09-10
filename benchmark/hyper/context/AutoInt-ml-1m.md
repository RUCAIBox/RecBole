# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [AutoInt](https://recbole.io/docs/user_guide/model/context/autoint.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-3,5e-3]
    dropout_prob choice [0.0,0.1]
    attention_size choice [8,16,32]
    mlp_hidden_size choice ['[64,64,64]','[128,128,128]','[256,256,256]']
  ```

- **Best parameters**:

  ```
    attention_size: 8
    dropout_prob: 0.1
    learning_rate: 0.005
    mlp_hidden_size: [256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    attention_size:8, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9004    logloss : 0.3132
    Test result:
    auc : 0.8987    logloss : 0.3174

    attention_size:32, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.8996    logloss : 0.3145
    Test result:
    auc : 0.8971    logloss : 0.32

    attention_size:16, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9002    logloss : 0.3159
    Test result:
    auc : 0.8978    logloss : 0.3207

    attention_size:8, dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8974    logloss : 0.3198
    Test result:
    auc : 0.8955    logloss : 0.3238

    attention_size:32, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9001    logloss : 0.3118
    Test result:
    auc : 0.8988    logloss : 0.3154

    attention_size:16, dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8982    logloss : 0.3173
    Test result:
    auc : 0.8975    logloss : 0.3196

    attention_size:32, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8982    logloss : 0.3171
    Test result:
    auc : 0.8969    logloss : 0.32

    attention_size:16, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9004    logloss : 0.3163
    Test result:
    auc : 0.8985    logloss : 0.3205

    attention_size:8, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8987    logloss : 0.3157
    Test result:
    auc : 0.8975    logloss : 0.3185

    attention_size:8, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.8991    logloss : 0.3149
    Test result:
    auc : 0.8974    logloss : 0.3182
  ```