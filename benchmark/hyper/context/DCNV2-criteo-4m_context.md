# Context Recommendation

- **Dataset**: [Criteo](../../md/criteo-4m_context.md)

- **Model**: [DCNV2](https://recbole.io/docs/user_guide/model/context/dcnv2.html)

- **Time cost**: 652.88s/trial

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
   'cross_layer_num': 2
   'dropout_prob': 0.1
   'learning_rate': 0.001
   'mlp_hidden_size': '[256,256]'
   'reg_weight': 5
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  cross_layer_num:2, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[512,512], reg_weight:1
    Valid result:
    auc : 0.7929    logloss : 0.4528
    Test result:
    auc : 0.7949    logloss : 0.4519
    
    cross_layer_num:3, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[768,768], reg_weight:5
    Valid result:
    auc : 0.7932    logloss : 0.4518
    Test result:
    auc : 0.7949    logloss : 0.4512
    
    cross_layer_num:4, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[512,512], reg_weight:5
    Valid result:
    auc : 0.793    logloss : 0.4521
    Test result:
    auc : 0.7947    logloss : 0.4514
    
    cross_layer_num:2, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[512,512], reg_weight:1
    Valid result:
    auc : 0.7922    logloss : 0.453
    Test result:
    auc : 0.7944    logloss : 0.4522
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[512,512], reg_weight:5
    Valid result:
    auc : 0.79    logloss : 0.4546
    Test result:
    auc : 0.7921    logloss : 0.4535
    
    cross_layer_num:3, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[768,768], reg_weight:1
    Valid result:
    auc : 0.7926    logloss : 0.4522
    Test result:
    auc : 0.7943    logloss : 0.4517
    
    cross_layer_num:4, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[768,768], reg_weight:5
    Valid result:
    auc : 0.7901    logloss : 0.4543
    Test result:
    auc : 0.7921    logloss : 0.4534
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[768,768], reg_weight:1
    Valid result:
    auc : 0.7904    logloss : 0.4545
    Test result:
    auc : 0.7923    logloss : 0.4535
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[256,256], reg_weight:5
    Valid result:
    auc : 0.7891    logloss : 0.4557
    Test result:
    auc : 0.7913    logloss : 0.4548
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256], reg_weight:5
    Valid result:
    auc : 0.7937    logloss : 0.4515
    Test result:
    auc : 0.7955    logloss : 0.4508
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[768,768], reg_weight:1
    Valid result:
    auc : 0.7915    logloss : 0.4535
    Test result:
    auc : 0.7936    logloss : 0.4527
    
    cross_layer_num:4, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[256,256], reg_weight:2
    Valid result:
    auc : 0.7897    logloss : 0.4548
    Test result:
    auc : 0.792    logloss : 0.4538
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[256,256], reg_weight:5
    Valid result:
    auc : 0.7915    logloss : 0.4534
    Test result:
    auc : 0.7936    logloss : 0.4527
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256], reg_weight:1
    Valid result:
    auc : 0.7936    logloss : 0.4516
    Test result:
    auc : 0.7955    logloss : 0.4508
    
    cross_layer_num:3, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[768,768], reg_weight:5
    Valid result:
    auc : 0.7901    logloss : 0.4542
    Test result:
    auc : 0.7922    logloss : 0.4534
    
    cross_layer_num:2, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[256,256], reg_weight:1
    Valid result:
    auc : 0.7923    logloss : 0.4528
    Test result:
    auc : 0.7942    logloss : 0.4521
    
    cross_layer_num:4, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[768,768], reg_weight:1
    Valid result:
    auc : 0.7927    logloss : 0.4525
    Test result:
    auc : 0.7948    logloss : 0.4517
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[1024,1024], reg_weight:5
    Valid result:
    auc : 0.7911    logloss : 0.4538
    Test result:
    auc : 0.793    logloss : 0.453
    
    cross_layer_num:2, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[768,768], reg_weight:5
    Valid result:
    auc : 0.7922    logloss : 0.4532
    Test result:
    auc : 0.7943    logloss : 0.4524
    
    cross_layer_num:4, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[256,256], reg_weight:1
    Valid result:
    auc : 0.7911    logloss : 0.4538
    Test result:
    auc : 0.7928    logloss : 0.4532
  ```

- **Logging Result**:

  ```yaml
    9%|█▏           | 20/216 [3:37:37<35:32:45, 652.88s/trial, best loss: -0.7937]
    best params:  {'cross_layer_num': 2, 'dropout_prob': 0.1, 'learning_rate': 0.001, 'mlp_hidden_size': '[256,256]', 'reg_weight': 5}
    best result: 
    {'model': 'DCNV2', 'best_valid_score': 0.7937, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7937), ('logloss', 0.4515)]), 'test_result': OrderedDict([('auc', 0.7955), ('logloss', 0.4508)])}
  ```
