# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [AutoInt](https://recbole.io/docs/user_guide/model/context/autoint.html)

- **Time cost**: 987.87s/trial

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
    learning_rate: 0.001
    mlp_hidden_size: [256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    attention_size:32, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.793    logloss : 0.4527
    Test result:
    auc : 0.7949    logloss : 0.4518

    attention_size:8, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7916    logloss : 0.4536
    Test result:
    auc : 0.7933    logloss : 0.4529

    attention_size:16, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7912    logloss : 0.4536
    Test result:
    auc : 0.7932    logloss : 0.4527

    attention_size:16, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7916    logloss : 0.4533
    Test result:
    auc : 0.7935    logloss : 0.4525

    attention_size:8, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7926    logloss : 0.4525
    Test result:
    auc : 0.7947    logloss : 0.4516

    attention_size:32, dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7912    logloss : 0.4539
    Test result:
    auc : 0.7932    logloss : 0.4531

    attention_size:8, dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7905    logloss : 0.4541
    Test result:
    auc : 0.7925    logloss : 0.4532

    attention_size:8, dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7906    logloss : 0.4544
    Test result:
    auc : 0.7925    logloss : 0.4536

    attention_size:16, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7927    logloss : 0.4524
    Test result:
    auc : 0.7946    logloss : 0.4516

    attention_size:8, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7916    logloss : 0.4536
    Test result:
    auc : 0.7933    logloss : 0.4529
  ```

- **Logging Result**:

  ```yaml
    28%|██▊       | 10/36 [2:21:15<6:07:16, 847.57s/trial, best loss: -0.793]
    best params:  {'attention_size': 32, 'dropout_prob': 0.1, 'learning_rate': 0.001, 'mlp_hidden_size': '[256,256,256]'}
    best result: 
    {'model': 'AutoInt', 'best_valid_score': 0.793, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.793), ('logloss', 0.4527)]), 'test_result': OrderedDict([('auc', 0.7949), ('logloss', 0.4518)])}
  ```
