# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [DIEN](https://recbole.io/docs/user_guide/model/context/dien.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4, 1e-3]
    mlp_hidden_size choice ['[128,128,128]','[256,256,256]']
    dropout_prob choice [0.0, 0.1]
  ```

- **Best parameters**:

  ```
    dropout_prob': 0.1
    learning_rate: 0.001
    mlp_hidden_size: '[256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9421    logloss : 0.2332
    Test result:
    auc : 0.9306    logloss : 0.24

    dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.9413    logloss : 0.2607
    Test result:
    auc : 0.9313    logloss : 0.2657

    dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.9509    logloss : 0.2838
    Test result:
    auc : 0.9395    logloss : 0.2902

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.9478    logloss : 0.2697
    Test result:
    auc : 0.9362    logloss : 0.2788

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.9537    logloss : 0.2635
    Test result:
    auc : 0.9425    logloss : 0.2684

    dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9521    logloss : 0.263
    Test result:
    auc : 0.9413    logloss : 0.2703

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9371    logloss : 0.2373
    Test result:
    auc : 0.9248    logloss : 0.2433

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.9538    logloss : 0.2426
    Test result:
    auc : 0.9429    logloss : 0.248
  ```

- **Logging Result**:

  ```yaml
    100%|██████████| 8/8 [45:15:56<00:00, 20369.56s/trial, best loss: -0.9538]
    best params:  {'dropout_prob': 0.1, 'learning_rate': 0.001, 'mlp_hidden_size': '[256,256,256]'}
    best result: 
    {'model': 'DIEN', 'best_valid_score': 0.9538, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.9538), ('logloss', 0.2426)]), 'test_result': OrderedDict([('auc', 0.9429), ('logloss', 0.248)])}
  ```