# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [WideDeep](https://recbole.io/docs/user_guide/model/context/widedeep.html)

- **Time cost**: 548.90s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-4,1e-3,5e-3,1e-2]
    dropout_prob choice [0.0,0.2]
    mlp_hidden_size choice ['[64,64,64]','[128,128,128]','[256,256,256]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.0
    learning_rate: 0.001
    mlp_hidden_size: [256,256,256]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.2, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7828    logloss : 0.4634
    Test result:
    auc : 0.7849    logloss : 0.4621

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7844    logloss : 0.4624
    Test result:
    auc : 0.7861    logloss : 0.4615

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7846    logloss : 0.4617
    Test result:
    auc : 0.7866    logloss : 0.4606

    dropout_prob:0.0, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7846    logloss : 0.4626
    Test result:
    auc : 0.7864    logloss : 0.4617

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7934    logloss : 0.4516
    Test result:
    auc : 0.7951    logloss : 0.451

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.79    logloss : 0.4545
    Test result:
    auc : 0.792    logloss : 0.4536

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7909    logloss : 0.4543
    Test result:
    auc : 0.7929    logloss : 0.4536

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7922    logloss : 0.453
    Test result:
    auc : 0.7938    logloss : 0.4524

    dropout_prob:0.2, learning_rate:0.01, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7838    logloss : 0.4625
    Test result:
    auc : 0.7857    logloss : 0.4615

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7917    logloss : 0.4533
    Test result:
    auc : 0.7936    logloss : 0.4525

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7925    logloss : 0.4525
    Test result:
    auc : 0.7942    logloss : 0.4519

    dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.7927    logloss : 0.4525
    Test result:
    auc : 0.7945    logloss : 0.4517

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7942    logloss : 0.4509
    Test result:
    auc : 0.796    logloss : 0.4503

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7905    logloss : 0.4541
    Test result:
    auc : 0.7924    logloss : 0.4533

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7922    logloss : 0.4526
    Test result:
    auc : 0.7942    logloss : 0.4518

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7902    logloss : 0.4542
    Test result:
    auc : 0.7923    logloss : 0.4534

    dropout_prob:0.2, learning_rate:0.01, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7837    logloss : 0.4626
    Test result:
    auc : 0.7856    logloss : 0.4616

    dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[128,128,128]
    Valid result:
    auc : 0.791    logloss : 0.4536
    Test result:
    auc : 0.793    logloss : 0.4527

    dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7924    logloss : 0.4524
    Test result:
    auc : 0.7942    logloss : 0.4517

    dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7909    logloss : 0.4539
    Test result:
    auc : 0.7928    logloss : 0.4531

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7924    logloss : 0.4526
    Test result:
    auc : 0.7941    logloss : 0.4519

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[64,64,64]
    Valid result:
    auc : 0.7895    logloss : 0.4551
    Test result:
    auc : 0.7916    logloss : 0.4541

    dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[256,256,256]
    Valid result:
    auc : 0.7919    logloss : 0.4532
    Test result:
    auc : 0.7937    logloss : 0.4523
  ```

- **Logging Result**:

  ```yaml
    96%|█████████▌| 23/24 [3:30:30<09:09, 549.17s/trial, best loss: -0.7942]
    best params:  {'dropout_prob': 0.0, 'learning_rate': 0.001, 'mlp_hidden_size': '[256,256,256]'}
    best result: 
    {'model': 'WideDeep', 'best_valid_score': 0.7942, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7942), ('logloss', 0.4509)]), 'test_result': OrderedDict([('auc', 0.796), ('logloss', 0.4503)])}
  ```