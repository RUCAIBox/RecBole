# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [FM](https://recbole.io/docs/user_guide/model/context/fm.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,1e-4,2e-4,5e-4,1e-3,5e-3]
  ```

- **Best parameters**:

  ```
    learning_rate: 0.005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    learning_rate:0.0005
    Valid result:
    auc : 0.8914    logloss : 0.3349
    Test result:
    auc : 0.8893    logloss : 0.3385

    learning_rate:0.005
    Valid result:
    auc : 0.896    logloss : 0.3219
    Test result:
    auc : 0.8935    logloss : 0.3262

    learning_rate:0.001
    Valid result:
    auc : 0.8925    logloss : 0.333
    Test result:
    auc : 0.8904    logloss : 0.337

    learning_rate:0.0002
    Valid result:
    auc : 0.89    logloss : 0.3415
    Test result:
    auc : 0.8881    logloss : 0.3454

    learning_rate:0.0001
    Valid result:
    auc : 0.8895    logloss : 0.3402
    Test result:
    auc : 0.8876    logloss : 0.3441

    learning_rate:5e-05
    Valid result:
    auc : 0.887    logloss : 0.3327
    Test result:
    auc : 0.8847    logloss : 0.3358
  ```