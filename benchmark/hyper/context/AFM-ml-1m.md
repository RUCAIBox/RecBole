# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [AFM](https://recbole.io/docs/user_guide/model/context/afm.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,1e-4,5e-4]
    dropout_prob choice [0.0,0.1]
    attention_size choice [20,30]
    reg_weight choice [2,5]
  ```

- **Best parameters**:

  ```
    attention_size: 30
    dropout_prob: 0.1
    learning_rate: 0.0005
    reg_weight: 5
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    attention_size:30, dropout_prob:0.0, learning_rate:0.0005, reg_weight:5
    Valid result:
    auc : 0.8944    logloss : 0.331
    Test result:
    auc : 0.8934    logloss : 0.3336

    attention_size:20, dropout_prob:0.1, learning_rate:0.0005, reg_weight:5
    Valid result:
    auc : 0.8983    logloss : 0.3247
    Test result:
    auc : 0.8973    logloss : 0.327

    attention_size:30, dropout_prob:0.1, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.8908    logloss : 0.3227
    Test result:
    auc : 0.8884    logloss : 0.3265

    attention_size:20, dropout_prob:0.0, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.8804    logloss : 0.3383
    Test result:
    auc : 0.8772    logloss : 0.3427

    attention_size:20, dropout_prob:0.1, learning_rate:0.0001, reg_weight:2
    Valid result:
    auc : 0.8896    logloss : 0.3244
    Test result:
    auc : 0.8872    logloss : 0.3282

    attention_size:30, dropout_prob:0.1, learning_rate:5e-05, reg_weight:2
    Valid result:
    auc : 0.8776    logloss : 0.3391
    Test result:
    auc : 0.8738    logloss : 0.3439

    attention_size:20, dropout_prob:0.1, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.8771    logloss : 0.3396
    Test result:
    auc : 0.8734    logloss : 0.3444

    attention_size:20, dropout_prob:0.1, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.8896    logloss : 0.3244
    Test result:
    auc : 0.8872    logloss : 0.3282

    attention_size:20, dropout_prob:0.0, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.893    logloss : 0.3328
    Test result:
    auc : 0.8919    logloss : 0.3349

    attention_size:20, dropout_prob:0.0, learning_rate:0.0005, reg_weight:5
    Valid result:
    auc : 0.893    logloss : 0.3348
    Test result:
    auc : 0.8919    logloss : 0.3373

    attention_size:20, dropout_prob:0.0, learning_rate:5e-05, reg_weight:2
    Valid result:
    auc : 0.875    logloss : 0.3428
    Test result:
    auc : 0.871    logloss : 0.3479

    attention_size:30, dropout_prob:0.1, learning_rate:0.0005, reg_weight:5
    Valid result:
    auc : 0.8991    logloss : 0.3216
    Test result:
    auc : 0.8981    logloss : 0.3238

    attention_size:30, dropout_prob:0.0, learning_rate:0.0001, reg_weight:2
    Valid result:
    auc : 0.8808    logloss : 0.3377
    Test result:
    auc : 0.8775    logloss : 0.3421

    attention_size:20, dropout_prob:0.1, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.8983    logloss : 0.3251
    Test result:
    auc : 0.8973    logloss : 0.3273

    attention_size:20, dropout_prob:0.1, learning_rate:5e-05, reg_weight:2
    Valid result:
    auc : 0.8771    logloss : 0.3396
    Test result:
    auc : 0.8734    logloss : 0.3444

    attention_size:30, dropout_prob:0.0, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.8753    logloss : 0.343
    Test result:
    auc : 0.8716    logloss : 0.3479

    attention_size:30, dropout_prob:0.1, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.8776    logloss : 0.3391
    Test result:
    auc : 0.8739    logloss : 0.3439

    attention_size:30, dropout_prob:0.0, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.8945    logloss : 0.3296
    Test result:
    auc : 0.8934    logloss : 0.3324

    attention_size:20, dropout_prob:0.0, learning_rate:5e-05, reg_weight:5
    Valid result:
    auc : 0.875    logloss : 0.3428
    Test result:
    auc : 0.871    logloss : 0.3479

    attention_size:30, dropout_prob:0.1, learning_rate:0.0005, reg_weight:2
    Valid result:
    auc : 0.8991    logloss : 0.3216
    Test result:
    auc : 0.8981    logloss : 0.3239

    attention_size:30, dropout_prob:0.0, learning_rate:0.0001, reg_weight:5
    Valid result:
    auc : 0.8808    logloss : 0.3376
    Test result:
    auc : 0.8775    logloss : 0.3421

    attention_size:30, dropout_prob:0.1, learning_rate:0.0001, reg_weight:2
    Valid result:
    auc : 0.8908    logloss : 0.3227
    Test result:
    auc : 0.8884    logloss : 0.3266
  ```