# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [FwFM](https://recbole.io/docs/user_guide/model/context/fwfm.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4,5e-4,1e-3,5e-3,1e-2]
    dropout_prob choice [0.0,0.2,0.4]
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.2
    learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.0, learning_rate:0.001
    Valid result:
    auc : 0.8739    logloss : 0.3443
    Test result:
    auc : 0.8698    logloss : 0.349

    dropout_prob:0.0, learning_rate:0.0005
    Valid result:
    auc : 0.8738    logloss : 0.3443
    Test result:
    auc : 0.8696    logloss : 0.3493

    dropout_prob:0.0, learning_rate:0.005
    Valid result:
    auc : 0.8734    logloss : 0.3447
    Test result:
    auc : 0.869    logloss : 0.3499

    dropout_prob:0.4, learning_rate:0.001
    Valid result:
    auc : 0.8739    logloss : 0.3441
    Test result:
    auc : 0.8696    logloss : 0.3491

    dropout_prob:0.2, learning_rate:0.01
    Valid result:
    auc : 0.8725    logloss : 0.3461
    Test result:
    auc : 0.8673    logloss : 0.3522

    dropout_prob:0.0, learning_rate:0.01
    Valid result:
    auc : 0.872    logloss : 0.3465
    Test result:
    auc : 0.8669    logloss : 0.3527

    dropout_prob:0.4, learning_rate:0.005
    Valid result:
    auc : 0.8735    logloss : 0.3448
    Test result:
    auc : 0.8692    logloss : 0.3498

    dropout_prob:0.4, learning_rate:0.0001
    Valid result:
    auc : 0.8735    logloss : 0.3451
    Test result:
    auc : 0.8694    logloss : 0.3498

    dropout_prob:0.2, learning_rate:0.0001
    Valid result:
    auc : 0.8735    logloss : 0.3451
    Test result:
    auc : 0.8694    logloss : 0.3498

    dropout_prob:0.4, learning_rate:0.01
    Valid result:
    auc : 0.8725    logloss : 0.3458
    Test result:
    auc : 0.8675    logloss : 0.3519
  ```