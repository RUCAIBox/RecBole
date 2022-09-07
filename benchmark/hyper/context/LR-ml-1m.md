# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [LR](https://recbole.io/docs/user_guide/model/context/lr.html)

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
    learning_rate:0.001
    Valid result:
    auc : 0.8739    logloss : 0.3442
    Test result:
    auc : 0.8698    logloss : 0.349

    learning_rate:5e-05
    Valid result:
    auc : 0.87    logloss : 0.3536
    Test result:
    auc : 0.866    logloss : 0.3578

    learning_rate:0.0002
    Valid result:
    auc : 0.8737    logloss : 0.3445
    Test result:
    auc : 0.8696    logloss : 0.3493

    learning_rate:0.0001
    Valid result:
    auc : 0.8735    logloss : 0.3451
    Test result:
    auc : 0.8694    logloss : 0.3498

    learning_rate:0.0005
    Valid result:
    auc : 0.8738    logloss : 0.3443
    Test result:
    auc : 0.8697    logloss : 0.3491

    learning_rate:0.005
    Valid result:
    auc : 0.874    logloss : 0.3439
    Test result:
    auc : 0.8699    logloss : 0.3488
  ```