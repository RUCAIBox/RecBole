# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [FNN](https://recbole.io/docs/user_guide/model/context/fnn.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-4,1e-3,3e-3,5e-3]
    dropout_prob choice [0.0,0.1]
    mlp_hidden_size choice ['[128,256,128]','[128,128,128]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.1
    learning_rate: 0.003
    mlp_hidden_size: [128,256,128]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.8885    logloss : 0.3286
    Test result:
    auc : 0.8875    logloss : 0.3309

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8892    logloss : 0.3272
    Test result:
    auc : 0.8882    logloss : 0.3299

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.891    logloss : 0.325
    Test result:
    auc : 0.8903    logloss : 0.3272

    dropout_prob:0.0, learning_rate:0.003, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8906    logloss : 0.3269
    Test result:
    auc : 0.8888    logloss : 0.3312

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.8883    logloss : 0.328
    Test result:
    auc : 0.8869    logloss : 0.3307

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8914    logloss : 0.3242
    Test result:
    auc : 0.8906    logloss : 0.3263

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8892    logloss : 0.3289
    Test result:
    auc : 0.8884    logloss : 0.3312

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.891    logloss : 0.3301
    Test result:
    auc : 0.8891    logloss : 0.3338

    dropout_prob:0.1, learning_rate:0.003, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8901    logloss : 0.3272
    Test result:
    auc : 0.8888    logloss : 0.3306

    dropout_prob:0.0, learning_rate:0.003, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.8909    logloss : 0.3237
    Test result:
    auc : 0.8896    logloss : 0.3264

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.8911    logloss : 0.3259
    Test result:
    auc : 0.8887    logloss : 0.3299

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8728    logloss : 0.3449
    Test result:
    auc : 0.8685    logloss : 0.3499

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.8911    logloss : 0.3275
    Test result:
    auc : 0.8894    logloss : 0.3316

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8729    logloss : 0.3447
    Test result:
    auc : 0.8686    logloss : 0.3498

    dropout_prob:0.1, learning_rate:0.003, mlp_hidden_size:[128,256,128]
    Valid result:
    auc : 0.8916    logloss : 0.3231
    Test result:
    auc : 0.8903    logloss : 0.3267

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.8904    logloss : 0.3255
    Test result:
    auc : 0.8895    logloss : 0.328
  ```