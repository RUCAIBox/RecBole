# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [DCN](https://recbole.io/docs/user_guide/model/context/dcn.html)

- **Time cost**: 602.62s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4,5e-4,1e-3,5e-3,6e-3]
    mlp_hidden_size choice ['[128,128,128]','[256,256,256]','[512,512,512]','[1024,1024,1024]']
    reg_weight choice [1,2,5]
    cross_layer_num choice [6]
    dropout_prob choice [0.1,0.2]
  ```

- **Best parameters**:

  ```
    cross_layer_num: 6
    dropout_prob: 0.1
    learning_rate: 0.005
    mlp_hidden_size: '[512,512,512]
    reg_weight: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[256,256,256], reg_weight:2
    Valid result:
    auc : 0.7924    logloss : 0.4531
    Test result:
    auc : 0.7944    logloss : 0.4521

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[512,512,512], reg_weight:1
    Valid result:
    auc : 0.7895    logloss : 0.455
    Test result:
    auc : 0.7918    logloss : 0.4539

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[128,128,128], reg_weight:5
    Valid result:
    auc : 0.7893    logloss : 0.4554
    Test result:
    auc : 0.7914    logloss : 0.4545

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[1024,1024,1024], reg_weight:1
    Valid result:
    auc : 0.7919    logloss : 0.4541
    Test result:
    auc : 0.794    logloss : 0.4532

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:5
    Valid result:
    auc : 0.7861    logloss : 0.4579
    Test result:
    auc : 0.7879    logloss : 0.4571

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[512,512,512], reg_weight:1
    Valid result:
    auc : 0.7936    logloss : 0.4519
    Test result:
    auc : 0.7953    logloss : 0.4513

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:1
    Valid result:
    auc : 0.7907    logloss : 0.454
    Test result:
    auc : 0.7928    logloss : 0.4532

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:1
    Valid result:
    auc : 0.791    logloss : 0.4538
    Test result:
    auc : 0.7931    logloss : 0.453

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:2
    Valid result:
    auc : 0.7856    logloss : 0.4583
    Test result:
    auc : 0.7877    logloss : 0.4574

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[256,256,256], reg_weight:1
    Valid result:
    auc : 0.7894    logloss : 0.455
    Test result:
    auc : 0.7914    logloss : 0.4542

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:1
    Valid result:
    auc : 0.7861    logloss : 0.4579
    Test result:
    auc : 0.7879    logloss : 0.4571

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:2
    Valid result:
    auc : 0.7904    logloss : 0.4542
    Test result:
    auc : 0.7925    logloss : 0.4534

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[256,256,256], reg_weight:5
    Valid result:
    auc : 0.7927    logloss : 0.4525
    Test result:
    auc : 0.7945    logloss : 0.4517

    cross_layer_num:6, dropout_prob:0.2, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1
    Valid result:
    auc : 0.7925    logloss : 0.4527
    Test result:
    auc : 0.7947    logloss : 0.4517

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[1024,1024,1024], reg_weight:1
    Valid result:
    auc : 0.7856    logloss : 0.4584
    Test result:
    auc : 0.7876    logloss : 0.4573

    cross_layer_num:6, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:5
    Valid result:
    auc : 0.793    logloss : 0.4521
    Test result:
    auc : 0.7951    logloss : 0.4512
  ```

- **Logging Result**:

  ```yaml
    13%|█▎        | 16/120 [2:45:03<17:52:49, 618.94s/trial, best loss: -0.7936]
    best params:  {'cross_layer_num': 6, 'dropout_prob': 0.1, 'learning_rate': 0.005, 'mlp_hidden_size': '[512,512,512]', 'reg_weight': 1}
    best result: 
    {'model': 'DCN', 'best_valid_score': 0.7936, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7936), ('logloss', 0.4519)]), 'test_result': OrderedDict([('auc', 0.7953), ('logloss', 0.4513)])}
  ```
