# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [NGCF](https://recbole.io/docs/user_guide/model/general/ngcf.html)

- **Time cost**: 51413.37s/trial

- **Hyper-parameter searching**:

  ```yaml
  learning_rate choice [1e-4,5e-4]
  hidden_size_list choice ['[64,64,64]','[128,128,128]','[256,256,256]']
  node_dropout choice [0.0,0.1]
  message_dropout choice [0.0,0.1]
  reg_weight choice [1e-5]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 5e-4
  hidden_size_list: '[64,64,64]'
  node_dropout: 0.0
  message_dropout: 0.0
  reg_weight: 1e-5
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  hidden_size_list:[256,256,256], learning_rate:0.0001, message_dropout:0.0, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0528    mrr@10 : 0.0377    ndcg@10 : 0.0323    hit@10 : 0.1015    precision@10 : 0.0112
  Test result:
  recall@10 : 0.0535    mrr@10 : 0.0386    ndcg@10 : 0.0328    hit@10 : 0.1025    precision@10 : 0.0114
  
  hidden_size_list:[128,128,128], learning_rate:0.0001, message_dropout:0.1, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0704    mrr@10 : 0.0516    ndcg@10 : 0.0437    hit@10 : 0.1347    precision@10 : 0.0155
  Test result:
  recall@10 : 0.0723    mrr@10 : 0.052    ndcg@10 : 0.0444    hit@10 : 0.1363    precision@10 : 0.0157
  
  hidden_size_list:[128,128,128], learning_rate:0.0005, message_dropout:0.1, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0733    mrr@10 : 0.0533    ndcg@10 : 0.0455    hit@10 : 0.1393    precision@10 : 0.0162
  Test result:
  recall@10 : 0.0747    mrr@10 : 0.0542    ndcg@10 : 0.0462    hit@10 : 0.1406    precision@10 : 0.0163
  
  hidden_size_list:[64,64,64], learning_rate:0.0001, message_dropout:0.0, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0531    mrr@10 : 0.0379    ndcg@10 : 0.0325    hit@10 : 0.1016    precision@10 : 0.0112
  Test result:
  recall@10 : 0.0542    mrr@10 : 0.0384    ndcg@10 : 0.033    hit@10 : 0.1026    precision@10 : 0.0114
  
  hidden_size_list:[64,64,64], learning_rate:0.0001, message_dropout:0.1, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.07    mrr@10 : 0.0513    ndcg@10 : 0.0435    hit@10 : 0.1341    precision@10 : 0.0154
  Test result:
  recall@10 : 0.0711    mrr@10 : 0.0519    ndcg@10 : 0.0441    hit@10 : 0.1357    precision@10 : 0.0158
  
  hidden_size_list:[256,256,256], learning_rate:0.0001, message_dropout:0.1, node_dropout:0.1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0106    mrr@10 : 0.0075    ndcg@10 : 0.0061    hit@10 : 0.0249    precision@10 : 0.0026
  Test result:
  recall@10 : 0.0109    mrr@10 : 0.0078    ndcg@10 : 0.0064    hit@10 : 0.0254    precision@10 : 0.0027
  
  hidden_size_list:[64,64,64], learning_rate:0.0005, message_dropout:0.0, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0771    mrr@10 : 0.0556    ndcg@10 : 0.0478    hit@10 : 0.1447    precision@10 : 0.017
  Test result:
  recall@10 : 0.0794    mrr@10 : 0.0565    ndcg@10 : 0.049    hit@10 : 0.147    precision@10 : 0.0172
  
  hidden_size_list:[128,128,128], learning_rate:0.0001, message_dropout:0.0, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.053    mrr@10 : 0.0378    ndcg@10 : 0.0325    hit@10 : 0.1013    precision@10 : 0.0112
  Test result:
  recall@10 : 0.0537    mrr@10 : 0.0386    ndcg@10 : 0.0329    hit@10 : 0.1024    precision@10 : 0.0114
  
  hidden_size_list:[256,256,256], learning_rate:0.0005, message_dropout:0.1, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.072    mrr@10 : 0.0521    ndcg@10 : 0.0444    hit@10 : 0.1372    precision@10 : 0.0158
  Test result:
  recall@10 : 0.0736    mrr@10 : 0.0533    ndcg@10 : 0.0456    hit@10 : 0.1391    precision@10 : 0.0161
  
  hidden_size_list:[256,256,256], learning_rate:0.0005, message_dropout:0.1, node_dropout:0.1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0719    mrr@10 : 0.0521    ndcg@10 : 0.0444    hit@10 : 0.1368    precision@10 : 0.0158
  Test result:
  recall@10 : 0.0738    mrr@10 : 0.0535    ndcg@10 : 0.0456    hit@10 : 0.1387    precision@10 : 0.0161
  
  hidden_size_list:[256,256,256], learning_rate:0.0005, message_dropout:0.0, node_dropout:0.1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0754    mrr@10 : 0.0546    ndcg@10 : 0.0466    hit@10 : 0.1423    precision@10 : 0.0165
  Test result:
  recall@10 : 0.0775    mrr@10 : 0.0554    ndcg@10 : 0.0478    hit@10 : 0.145    precision@10 : 0.0169
  
  hidden_size_list:[64,64,64], learning_rate:0.0005, message_dropout:0.0, node_dropout:0.1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0745    mrr@10 : 0.0539    ndcg@10 : 0.0462    hit@10 : 0.1413    precision@10 : 0.0164
  Test result:
  recall@10 : 0.0762    mrr@10 : 0.0546    ndcg@10 : 0.0469    hit@10 : 0.1427    precision@10 : 0.0166
  
  hidden_size_list:[256,256,256], learning_rate:0.0005, message_dropout:0.0, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0757    mrr@10 : 0.0544    ndcg@10 : 0.0467    hit@10 : 0.1424    precision@10 : 0.0165
  Test result:
  recall@10 : 0.0768    mrr@10 : 0.0554    ndcg@10 : 0.0475    hit@10 : 0.1439    precision@10 : 0.0168
  
  hidden_size_list:[128,128,128], learning_rate:0.0005, message_dropout:0.0, node_dropout:0.1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0741    mrr@10 : 0.0534    ndcg@10 : 0.0457    hit@10 : 0.14    precision@10 : 0.0162
  Test result:
  recall@10 : 0.0762    mrr@10 : 0.0551    ndcg@10 : 0.0471    hit@10 : 0.1433    precision@10 : 0.0166
  
  hidden_size_list:[128,128,128], learning_rate:0.0005, message_dropout:0.0, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0753    mrr@10 : 0.0544    ndcg@10 : 0.0465    hit@10 : 0.1422    precision@10 : 0.0164
  Test result:
  recall@10 : 0.0761    mrr@10 : 0.0546    ndcg@10 : 0.0469    hit@10 : 0.1427    precision@10 : 0.0167
  
  hidden_size_list:[64,64,64], learning_rate:0.0005, message_dropout:0.1, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.071    mrr@10 : 0.052    ndcg@10 : 0.0443    hit@10 : 0.1354    precision@10 : 0.0158
  Test result:
  recall@10 : 0.0728    mrr@10 : 0.0523    ndcg@10 : 0.0447    hit@10 : 0.1385    precision@10 : 0.0161
  
  hidden_size_list:[128,128,128], learning_rate:0.0001, message_dropout:0.0, node_dropout:0.1, reg_weight:1e-05
  Valid result:
  recall@10 : 0.0529    mrr@10 : 0.0378    ndcg@10 : 0.0324    hit@10 : 0.1012    precision@10 : 0.0111
  Test result:
  recall@10 : 0.0539    mrr@10 : 0.0385    ndcg@10 : 0.033    hit@10 : 0.1022    precision@10 : 0.0113
  ```
  
- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  71%|███████   | 17/24 [242:47:07<99:58:13, 51413.37s/trial, best loss: -0.0478]
  best params:  {'hidden_size_list': '[64,64,64]', 'learning_rate': 0.0005, 'message_dropout': 0.0, 'node_dropout': 0.0, 'reg_weight': 1e-05}
  best result: 
  {'model': 'NGCF', 'best_valid_score': 0.0478, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0771), ('mrr@10', 0.0556), ('ndcg@10', 0.0478), ('hit@10', 0.1447), ('precision@10', 0.017)]), 'test_result': OrderedDict([('recall@10', 0.0794), ('mrr@10', 0.0565), ('ndcg@10', 0.049), ('hit@10', 0.147), ('precision@10', 0.0172)])}
  ```
