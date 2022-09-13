# Context Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_context.md)

- **Model**: [DCNV2](https://recbole.io/docs/user_guide/model/context/dcnv2.html)

- **Time cost**: 72.87s/trial

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
   'dropout_prob': 0.2
   'learning_rate': 0.005
   'mlp_hidden_size': '[256,256]'
   'reg_weight': 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  cross_layer_num:3, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[512,512], reg_weight:1
  Valid result:
  auc : 0.894    logloss : 0.3324
  Test result:
  auc : 0.8925    logloss : 0.3357

  cross_layer_num:4, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[768,768], reg_weight:5
  Valid result:
  auc : 0.8991    logloss : 0.3178
  Test result:
  auc : 0.8983    logloss : 0.3196

  cross_layer_num:4, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[1024,1024], reg_weight:2
  Valid result:
  auc : 0.8975    logloss : 0.3325
  Test result:
  auc : 0.8962    logloss : 0.3343
  
  cross_layer_num:2, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[512,512], reg_weight:2
  Valid result:
  auc : 0.8971    logloss : 0.3264
  Test result:
  auc : 0.8967    logloss : 0.3267
  
  cross_layer_num:2, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[512,512], reg_weight:1
  Valid result:
  auc : 0.8966    logloss : 0.3267
  Test result:
  auc : 0.8952    logloss : 0.3293
  
  cross_layer_num:4, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[512,512], reg_weight:1
  Valid result:
  auc : 0.8989    logloss : 0.3198
  Test result:
  auc : 0.8964    logloss : 0.3243
  
  cross_layer_num:3, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[1024,1024], reg_weight:2
  Valid result:
  auc : 0.8992    logloss : 0.3214
  Test result:
  auc : 0.8984    logloss : 0.3227
  
  cross_layer_num:2, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[512,512], reg_weight:1
  Valid result:
  auc : 0.8987    logloss : 0.3165
  Test result:
  auc : 0.8964    logloss : 0.3208
  
  cross_layer_num:2, dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[512,512], reg_weight:1
  Valid result:
  auc : 0.8934    logloss : 0.3363
  Test result:
  auc : 0.8918    logloss : 0.3392
  
  cross_layer_num:2, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[256,256], reg_weight:1
  Valid result:
  auc : 0.9008    logloss : 0.3096
  Test result:
  auc : 0.8998    logloss : 0.3119
  
  cross_layer_num:3, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[256,256], reg_weight:1
  Valid result:
  auc : 0.8933    logloss : 0.3297
  Test result:
  auc : 0.8918    logloss : 0.3323
  
  cross_layer_num:4, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[1024,1024], reg_weight:5
  Valid result:
  auc : 0.8926    logloss : 0.3316
  Test result:
  auc : 0.891    logloss : 0.3352
  
  cross_layer_num:2, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[512,512], reg_weight:2
  Valid result:
  auc : 0.8956    logloss : 0.3274
  Test result:
  auc : 0.8937    logloss : 0.3306
  
  cross_layer_num:3, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[768,768], reg_weight:2
  Valid result:
  auc : 0.8982    logloss : 0.3202
  Test result:
  auc : 0.8965    logloss : 0.3238
  
  cross_layer_num:2, dropout_prob:0.2, learning_rate:0.005, mlp_hidden_size:[1024,1024], reg_weight:5
  Valid result:
  auc : 0.9007    logloss : 0.3151
  Test result:
  auc : 0.8997    logloss : 0.3175
  
  cross_layer_num:3, dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[768,768], reg_weight:2
  Valid result:
  auc : 0.8942    logloss : 0.333
  Test result:
  auc : 0.8934    logloss : 0.3345
  
  cross_layer_num:2, dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[768,768], reg_weight:5
  Valid result:
  auc : 0.8986    logloss : 0.3215
  Test result:
  auc : 0.8974    logloss : 0.324
  
  cross_layer_num:2, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[768,768], reg_weight:2
  Valid result:
  auc : 0.8948    logloss : 0.3275
  Test result:
  auc : 0.8942    logloss : 0.3295
  
  cross_layer_num:4, dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[768,768], reg_weight:1
  Valid result:
  auc : 0.8989    logloss : 0.3198
  Test result:
  auc : 0.8981    logloss : 0.3212
  
  cross_layer_num:3, dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[1024,1024], reg_weight:5
  Valid result:
  auc : 0.8937    logloss : 0.3401
  Test result:
  auc : 0.8915    logloss : 0.3444
  ```

- **Logging Result**:

  ```yaml
  9%|█▌               | 20/216 [24:17<3:58:03, 72.87s/trial, best loss: -0.9008]
  best params:  {'cross_layer_num': 2, 'dropout_prob': 0.2, 'learning_rate': 0.005, 'mlp_hidden_size': '[256,256]', 'reg_weight': 1}
  best result: 
  {'model': 'DCNV2', 'best_valid_score': 0.9008, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.9008), ('logloss', 0.3096)]), 'test_result': OrderedDict([('auc', 0.8998), ('logloss', 0.3119)])}
  ```
