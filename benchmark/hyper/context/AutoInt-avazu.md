# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [AutoInt](https://recbole.io/docs/user_guide/model/context/autoint.html)

- **Time cost**: 244.25s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-3,5e-3]
    dropout_prob choice [0.0,0.1]
    attention_size choice [8,16,32]
    mlp_hidden_size choice ['[64,64,64]','[128,128,128]','[256,256,256]']
  ```

- **Best parameters**:

  ```
    attention_size: 32
    dropout_prob: 0.1
    learning_rate: 0.005
    mlp_hidden_size: [256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    attention_size:32, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7929    logloss : 0.3645
    Test result:
    auc : 0.7903    logloss : 0.3647

    attention_size:16, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.794    logloss : 0.3638
    Test result:
    auc : 0.7914    logloss : 0.3642

    attention_size:16, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7938    logloss : 0.3645
    Test result:
    auc : 0.791    logloss : 0.365

    attention_size:8, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7921    logloss : 0.3657
    Test result:
    auc : 0.7892    logloss : 0.3664

    attention_size:32, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7941    logloss : 0.364
    Test result:
    auc : 0.7916    logloss : 0.3642

    attention_size:32, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7941    logloss : 0.364
    Test result:
    auc : 0.7916    logloss : 0.3642

    attention_size:8, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7947    logloss : 0.3633
    Test result:
    auc : 0.7922    logloss : 0.3637

    attention_size:32, dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7923    logloss : 0.3649
    Test result:
    auc : 0.7891    logloss : 0.3656

    attention_size:8, dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7921    logloss : 0.3657
    Test result:
    auc : 0.7892    logloss : 0.3664

    attention_size:8, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7947    logloss : 0.3633
    Test result:
    auc : 0.7922    logloss : 0.3637

    attention_size:8, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.794    logloss : 0.3641
    Test result:
    auc : 0.791    logloss : 0.3647

    attention_size:32, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7923    logloss : 0.3649
    Test result:
    auc : 0.7891    logloss : 0.3656

    attention_size:32, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7913    logloss : 0.3655
    Test result:
    auc : 0.7881    logloss : 0.3662

    attention_size:32, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7948    logloss : 0.3631
    Test result:
    auc : 0.7922    logloss : 0.3634

    attention_size:8, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7947    logloss : 0.363
    Test result:
    auc : 0.7923    logloss : 0.3633

    attention_size:16, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7921    logloss : 0.3652
    Test result:
    auc : 0.7892    logloss : 0.3659

    attention_size:32, dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7929    logloss : 0.3644
    Test result:
    auc : 0.7898    logloss : 0.365

    attention_size:16, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7945    logloss : 0.3634
    Test result:
    auc : 0.792    logloss : 0.3637

    attention_size:8, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7934    logloss : 0.3641
    Test result:
    auc : 0.7902    logloss : 0.3649

    attention_size:16, dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7914    logloss : 0.3656
    Test result:
    auc : 0.7884    logloss : 0.3663

    attention_size:32, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.794    logloss : 0.3638
    Test result:
    auc : 0.7912    logloss : 0.3642

    attention_size:32, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.794    logloss : 0.3638
    Test result:
    auc : 0.7912    logloss : 0.3642

    attention_size:8, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.794    logloss : 0.3641
    Test result:
    auc : 0.791    logloss : 0.3647

    attention_size:16, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.794    logloss : 0.3638
    Test result:
    auc : 0.7914    logloss : 0.3642
  ```

- **Logging Result**:

  ```yaml
    67%|██████▋   | 24/36 [1:41:03<50:31, 252.66s/trial, best loss: -0.7948]
    best params:  {'attention_size': 32, 'dropout_prob': 0.1, 'learning_rate': 0.005, 'mlp_hidden_size': '[256,256,256]'}
    best result: 
    {'model': 'AutoInt', 'best_valid_score': 0.7948, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7948), ('logloss', 0.3631)]), 'test_result': OrderedDict([('auc', 0.7922), ('logloss', 0.3634)])}
  ```
