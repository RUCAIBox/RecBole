# Context Recommendation

- **Dataset**: [Avazu](../../md/avazu-2m_context.md)

- **Model**: [DCNV2](https://recbole.io/docs/user_guide/model/context/dcnv2.html)

- **Time cost**: 176.10s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005]
  mlp_hidden_size choice ['[256,256]','[512,512]','[768,768]','[1024, 1024]']
  cross_layer_num choice [2,3,4]
  dropout_prob choice [0.1,0.2]
  reg_weight choice [1,2,5]
  ```

- **Best parameters**:

  ```yaml
   'cross_layer_num': 4,
   'dropout_prob': 0.1,
   'learning_rate': 0.001,
   'mlp_hidden_size': '[256,256]',
   'reg_weight': 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  cross_layer_num:2, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[1024,1024], reg_weight:5
    Valid result:
    auc : 0.7939    logloss : 0.364
    Test result:
    auc : 0.7918    logloss : 0.3639
    
    cross_layer_num:3, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[512,512], reg_weight:5
    Valid result:
    auc : 0.796    logloss : 0.363
    Test result:
    auc : 0.7935    logloss : 0.3633
    
    cross_layer_num:2, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[1024,1024], reg_weight:1
    Valid result:
    auc : 0.7941    logloss : 0.3637
    Test result:
    auc : 0.792    logloss : 0.3637
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[512,512], reg_weight:2
    Valid result:
    auc : 0.7962    logloss : 0.3623
    Test result:
    auc : 0.7939    logloss : 0.3625
    
    cross_layer_num:3, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[512,512], reg_weight:2
    Valid result:
    auc : 0.7973    logloss : 0.3617
    Test result:
    auc : 0.7942    logloss : 0.3624
    
    cross_layer_num:2, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[256,256], reg_weight:1
    Valid result:
    auc : 0.7945    logloss : 0.3636
    Test result:
    auc : 0.7921    logloss : 0.3638
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[512,512], reg_weight:1
    Valid result:
    auc : 0.7968    logloss : 0.3629
    Test result:
    auc : 0.7943    logloss : 0.3633
    
    cross_layer_num:3, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[768,768], reg_weight:5
    Valid result:
    auc : 0.7958    logloss : 0.3629
    Test result:
    auc : 0.7926    logloss : 0.3638
    
    cross_layer_num:4, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256], reg_weight:2
    Valid result:
    auc : 0.7978    logloss : 0.3614
    Test result:
    auc : 0.7957    logloss : 0.3615
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[256,256], reg_weight:2
    Valid result:
    auc : 0.7951    logloss : 0.363
    Test result:
    auc : 0.7927    logloss : 0.3633
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[1024,1024], reg_weight:1
    Valid result:
    auc : 0.7957    logloss : 0.3624
    Test result:
    auc : 0.7934    logloss : 0.3628
    
    cross_layer_num:4, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[768,768], reg_weight:1
    Valid result:
    auc : 0.797    logloss : 0.3617
    Test result:
    auc : 0.7945    logloss : 0.362
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[512,512], reg_weight:5
    Valid result:
    auc : 0.7961    logloss : 0.3624
    Test result:
    auc : 0.7937    logloss : 0.3627
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[768,768], reg_weight:2
    Valid result:
    auc : 0.7968    logloss : 0.3619
    Test result:
    auc : 0.7945    logloss : 0.3623
    
    cross_layer_num:2, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[256,256], reg_weight:5
    Valid result:
    auc : 0.7931    logloss : 0.3643
    Test result:
    auc : 0.7905    logloss : 0.3649
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[256,256], reg_weight:2
    Valid result:
    auc : 0.7923    logloss : 0.3648
    Test result:
    auc : 0.7901    logloss : 0.3652
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[512,512], reg_weight:2
    Valid result:
    auc : 0.7969    logloss : 0.3628
    Test result:
    auc : 0.7945    logloss : 0.363
    
    cross_layer_num:3, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[256,256], reg_weight:2
    Valid result:
    auc : 0.7941    logloss : 0.3636
    Test result:
    auc : 0.7912    logloss : 0.3643
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[1024,1024], reg_weight:2
    Valid result:
    auc : 0.7942    logloss : 0.365
    Test result:
    auc : 0.7912    logloss : 0.3652
  ```

- **Logging Result**:

  ```yaml
      9%|█▍              | 19/216 [55:45<9:38:11, 176.10s/trial, best loss: -0.7978]
    best params:  {'cross_layer_num': 4, 'dropout_prob': 0.1, 'learning_rate': 0.001, 'mlp_hidden_size': '[256,256]', 'reg_weight': 2}
    best result: 
    {'model': 'DCNV2', 'best_valid_score': 0.7978, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7978), ('logloss', 0.3614)]), 'test_result': OrderedDict([('auc', 0.7957), ('logloss', 0.3615)])}
  ```
