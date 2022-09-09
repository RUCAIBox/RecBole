# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [DeepFM](https://recbole.io/docs/user_guide/model/context/deepfm.html)

- **Time cost**: 593.30s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-3,5e-3,1e-2]
    dropout_prob choice [0.0,0.1]
    mlp_hidden_size choice ['[128,128,128]','[256,256,256]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.0
    learning_rate: 0.001
    mlp_hidden_size: [256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.758    logloss : 0.5096
    Test result:
    auc : 0.7602    logloss : 0.5086

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.783    logloss : 0.4639
    Test result:
    auc : 0.7841    logloss : 0.4637

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7954    logloss : 0.4499
    Test result:
    auc : 0.7971    logloss : 0.4494

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7952    logloss : 0.4501
    Test result:
    auc : 0.7969    logloss : 0.4496

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7594    logloss : 0.505
    Test result:
    auc : 0.7609    logloss : 0.5045

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7578    logloss : 0.5065
    Test result:
    auc : 0.7601    logloss : 0.5052

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.781    logloss : 0.4672
    Test result:
    auc : 0.7828    logloss : 0.4664

    dropout_prob:0.1, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7551    logloss : 0.5156
    Test result:
    auc : 0.7577    logloss : 0.5139

    dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7804    logloss : 0.4684
    Test result:
    auc : 0.7819    logloss : 0.4677

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7951    logloss : 0.4503
    Test result:
    auc : 0.7968    logloss : 0.4497

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7835    logloss : 0.4624
    Test result:
    auc : 0.7848    logloss : 0.4621

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7956    logloss : 0.4498
    Test result:
    auc : 0.7973    logloss : 0.4493
  ```

- **Logging Result**:

  ```yaml
    100%|██████████| 12/12 [1:52:12<00:00, 561.04s/trial, best loss: -0.7956]
    best params:  {'dropout_prob': 0.0, 'learning_rate': 0.001, 'mlp_hidden_size': '[256,256,256]'}
    best result: 
    {'model': 'DeepFM', 'best_valid_score': 0.7956, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7956), ('logloss', 0.4498)]), 'test_result': OrderedDict([('auc', 0.7973), ('logloss', 0.4493)])}
  ```
