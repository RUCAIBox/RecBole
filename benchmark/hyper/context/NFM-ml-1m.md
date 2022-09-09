# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [NFM](https://recbole.io/docs/user_guide/model/context/nfm.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,8e-5,1e-4,5e-4,1e-3]
    dropout_prob choice [0.1,0.2,0.3]
    mlp_hidden_size choice ['[20,20,20]','[40,40,40]','[50,50,50]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.3
    learning_rate: 0.001
    mlp_hidden_size: [50,50,50]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.8849    logloss : 0.349
    Test result:
    auc : 0.8814    logloss : 0.3547

    dropout_prob:0.1, learning_rate:8e-05, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8775    logloss : 0.3447
    Test result:
    auc : 0.8748    logloss : 0.3488

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.89    logloss : 0.3345
    Test result:
    auc : 0.8889    logloss : 0.3369

    dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.878    logloss : 0.347
    Test result:
    auc : 0.8752    logloss : 0.3509

    dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8846    logloss : 0.3516
    Test result:
    auc : 0.8828    logloss : 0.3548

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8886    logloss : 0.337
    Test result:
    auc : 0.8863    logloss : 0.3411

    dropout_prob:0.3, learning_rate:8e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.8885    logloss : 0.3446
    Test result:
    auc : 0.8849    logloss : 0.3512

    dropout_prob:0.3, learning_rate:0.0005, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8923    logloss : 0.3328
    Test result:
    auc : 0.8898    logloss : 0.337

    dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.8784    logloss : 0.3655
    Test result:
    auc : 0.8767    logloss : 0.3692

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.8912    logloss : 0.3333
    Test result:
    auc : 0.8878    logloss : 0.3394

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.8885    logloss : 0.3436
    Test result:
    auc : 0.8851    logloss : 0.3508

    dropout_prob:0.3, learning_rate:5e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.8789    logloss : 0.3479
    Test result:
    auc : 0.876    logloss : 0.3523

    dropout_prob:0.1, learning_rate:8e-05, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.8768    logloss : 0.3439
    Test result:
    auc : 0.8752    logloss : 0.3464

    dropout_prob:0.2, learning_rate:8e-05, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.8835    logloss : 0.3557
    Test result:
    auc : 0.8826    logloss : 0.3584

    dropout_prob:0.3, learning_rate:0.001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8942    logloss : 0.325
    Test result:
    auc : 0.8915    logloss : 0.3295

    dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.8771    logloss : 0.3456
    Test result:
    auc : 0.874    logloss : 0.3496

    dropout_prob:0.2, learning_rate:8e-05, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8846    logloss : 0.3518
    Test result:
    auc : 0.8827    logloss : 0.3551

    dropout_prob:0.2, learning_rate:8e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.8844    logloss : 0.36
    Test result:
    auc : 0.8812    logloss : 0.3662

    dropout_prob:0.3, learning_rate:0.0001, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.8887    logloss : 0.3435
    Test result:
    auc : 0.8861    logloss : 0.3487

    dropout_prob:0.3, learning_rate:8e-05, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8892    logloss : 0.3392
    Test result:
    auc : 0.8863    logloss : 0.3438

    dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8777    logloss : 0.3453
    Test result:
    auc : 0.8751    logloss : 0.3493

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.8878    logloss : 0.3417
    Test result:
    auc : 0.8869    logloss : 0.3438

    dropout_prob:0.3, learning_rate:5e-05, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.8872    logloss : 0.3423
    Test result:
    auc : 0.8856    logloss : 0.3453

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.8888    logloss : 0.3331
    Test result:
    auc : 0.8864    logloss : 0.3371

    dropout_prob:0.1, learning_rate:5e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.8766    logloss : 0.3466
    Test result:
    auc : 0.8736    logloss : 0.3507
  ```