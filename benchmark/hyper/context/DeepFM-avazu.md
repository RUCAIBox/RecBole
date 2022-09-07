# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [DeepFM](https://recbole.io/docs/user_guide/model/context/deepfm.html)

- **Time cost**: 177.67s/trial

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
    mlp_hidden_size: [256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7972    logloss : 0.3621
    Test result:
    auc : 0.7952    logloss : 0.3622

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7948    logloss : 0.3641
    Test result:
    auc : 0.7928    logloss : 0.364

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.795    logloss : 0.3629
    Test result:
    auc : 0.7922    logloss : 0.3633

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7973    logloss : 0.3616
    Test result:
    auc : 0.7954    logloss : 0.3614

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7948    logloss : 0.363
    Test result:
    auc : 0.792    logloss : 0.3635

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.794    logloss : 0.3644
    Test result:
    auc : 0.7927    logloss : 0.364

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.794    logloss : 0.3636
    Test result:
    auc : 0.7913    logloss : 0.3641

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7975    logloss : 0.3617
    Test result:
    auc : 0.7959    logloss : 0.3614

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.796    logloss : 0.363
    Test result:
    auc : 0.7943    logloss : 0.3629

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7955    logloss : 0.3636
    Test result:
    auc : 0.7944    logloss : 0.363

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7943    logloss : 0.3634
    Test result:
    auc : 0.7915    logloss : 0.3639

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7974    logloss : 0.3617
    Test result:
    auc : 0.7953    logloss : 0.3616
  ```

- **Logging Result**:

  ```yaml
    100%|██████████| 12/12 [30:47<00:00, 153.93s/trial, best loss: -0.7975]
    best params:  {'dropout_prob': 0.1, 'learning_rate': 0.005, 'mlp_hidden_size': '[256,256,256]'}
    best result: 
    {'model': 'DeepFM', 'best_valid_score': 0.7975, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7975), ('logloss', 0.3617)]), 'test_result': OrderedDict([('auc', 0.7959), ('logloss', 0.3614)])}
  ```
