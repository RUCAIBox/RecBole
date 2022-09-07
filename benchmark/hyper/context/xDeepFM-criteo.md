# Context-aware Recommendation

- **Dataset**: [Criteo](../../md/criteo.md)

- **Model**: [xDeepFM](https://recbole.io/docs/user_guide/model/context/xdeepfm.html)

- **Time cost**: 1260.83s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4,1e-3,5e-3,6e-3]
    dropout_prob choice [0.0,0.1]
    mlp_hidden_size choice ['[128,128,128]','[256,256,256]','[512,512,512]']
    cin_layer_size choice ['[60,60,60]','[100,100,100]']
    reg_weight choice [1e-5,5e-4]
  ```

- **Best parameters**:

  ```
    cin_layer_size: [60,60,60]
    dropout_prob: 0.1
    learning_rate: 0.005
    mlp_hidden_size: [128,128,128]
    reg_weight: 1e-05
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.7935    logloss : 0.4523
    Test result:
    auc : 0.7958    logloss : 0.4511

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:0.0005
    Valid result:
    auc : 0.7915    logloss : 0.4535
    Test result:
    auc : 0.7935    logloss : 0.4527

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7849    logloss : 0.4594
    Test result:
    auc : 0.787    logloss : 0.4584

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:0.0005
    Valid result:
    auc : 0.7899    logloss : 0.4549
    Test result:
    auc : 0.792    logloss : 0.454

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.7844    logloss : 0.4598
    Test result:
    auc : 0.7865    logloss : 0.4588

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.7927    logloss : 0.4524
    Test result:
    auc : 0.7945    logloss : 0.4517

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.7939    logloss : 0.4518
    Test result:
    auc : 0.7958    logloss : 0.451

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[512,512,512], reg_weight:0.0005
    Valid result:
    auc : 0.7841    logloss : 0.4603
    Test result:
    auc : 0.786    logloss : 0.4593

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[512,512,512], reg_weight:0.0005
    Valid result:
    auc : 0.7881    logloss : 0.4562
    Test result:
    auc : 0.7902    logloss : 0.4553

    cin_layer_size:[60,60,60], dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[512,512,512], reg_weight:0.0005
    Valid result:
    auc : 0.7882    logloss : 0.4563
    Test result:
    auc : 0.7903    logloss : 0.4553

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7854    logloss : 0.459
    Test result:
    auc : 0.7875    logloss : 0.4581

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.786    logloss : 0.4583
    Test result:
    auc : 0.7879    logloss : 0.4576

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.7854    logloss : 0.4591
    Test result:
    auc : 0.7876    logloss : 0.4581

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.7953    logloss : 0.4506
    Test result:
    auc : 0.7968    logloss : 0.4501

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.006, mlp_hidden_size:[256,256,256], reg_weight:0.0005
    Valid result:
    auc : 0.7922    logloss : 0.4529
    Test result:
    auc : 0.794    logloss : 0.4522

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:0.0005
    Valid result:
    auc : 0.7851    logloss : 0.4591
    Test result:
    auc : 0.787    logloss : 0.4584

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[128,128,128], reg_weight:0.0005
    Valid result:
    auc : 0.785    logloss : 0.4592
    Test result:
    auc : 0.7868    logloss : 0.4585

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:0.0005
    Valid result:
    auc : 0.7843    logloss : 0.4599
    Test result:
    auc : 0.7864    logloss : 0.4591

    cin_layer_size:[60,60,60], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7849    logloss : 0.4593
    Test result:
    auc : 0.787    logloss : 0.4584

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[256,256,256], reg_weight:1e-05
    Valid result:
    auc : 0.7855    logloss : 0.4591
    Test result:
    auc : 0.7876    logloss : 0.4581

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.7936    logloss : 0.4522
    Test result:
    auc : 0.7951    logloss : 0.4516

    cin_layer_size:[100,100,100], dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256,256], reg_weight:0.0005
    Valid result:
    auc : 0.7881    logloss : 0.4563
    Test result:
    auc : 0.7903    logloss : 0.4553

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.006, mlp_hidden_size:[512,512,512], reg_weight:1e-05
    Valid result:
    auc : 0.7936    logloss : 0.4523
    Test result:
    auc : 0.7953    logloss : 0.4516

    cin_layer_size:[100,100,100], dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[128,128,128], reg_weight:1e-05
    Valid result:
    auc : 0.7931    logloss : 0.452
    Test result:
    auc : 0.7948    logloss : 0.4513
  ```

- **Logging Result**:

  ```yaml
    25%|██▌       | 24/96 [8:14:05<24:42:15, 1235.22s/trial, best loss: -0.7953]
    best params:  {'cin_layer_size': '[60,60,60]', 'dropout_prob': 0.1, 'learning_rate': 0.005, 'mlp_hidden_size': '[128,128,128]', 'reg_weight': 1e-05}
    best result: 
    {'model': 'xDeepFM', 'best_valid_score': 0.7953, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7953), ('logloss', 0.4506)]), 'test_result': OrderedDict([('auc', 0.7968), ('logloss', 0.4501)])}
  ```