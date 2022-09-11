# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [NGCF](https://recbole.io/docs/user_guide/model/general/ngcf.html)

- **Time cost**: 2875.00s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,5e-4,1e-3] 
  hidden_size_list choice ['[64,64,64]','[128,128,128]'] 
  node_dropout choice [0.0,0.1,0.2] 
  message_dropout choice [0.0,0.1,0.2] 
  reg_weight choice [1e-5,1e-3,1e-1]
  ```

- **Best parameters**:

  ```yaml
  hidden_size_list: [128,128,128]  
  learning_rate: 5e-4  
  message_dropout: 0.0  
  node_dropout: 0.0  
  reg_weight: 1e-5
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  hidden_size_list:'[128,128,128]', learning_rate:0.001, message_dropout:0.2, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.161     mrr@10 : 0.3542    ndcg@10 : 0.1966    hit@10 : 0.7016    precision@10 : 0.1467
  Test result:
  recall@10 : 0.1766    mrr@10 : 0.4191    ndcg@10 : 0.2378    hit@10 : 0.7258    precision@10 : 0.1774

  hidden_size_list:'[64,64,64]', learning_rate:0.001, message_dropout:0.1, node_dropout:0.0, reg_weight:0.1
  Valid result:
  recall@10 : 0.1603    mrr@10 : 0.353     ndcg@10 : 0.1969    hit@10 : 0.6955    precision@10 : 0.1465
  Test result:
  recall@10 : 0.175     mrr@10 : 0.415     ndcg@10 : 0.2373    hit@10 : 0.7224    precision@10 : 0.1776

  hidden_size_list:'[64,64,64]', learning_rate:0.0005, message_dropout:0.2, node_dropout:0.0, reg_weight:1e-05
  Valid result:
  recall@10 : 0.1566    mrr@10 : 0.3589    ndcg@10 : 0.1973    hit@10 : 0.6929    precision@10 : 0.1466
  Test result:
  recall@10 : 0.1708    mrr@10 : 0.417     ndcg@10 : 0.2375    hit@10 : 0.7159    precision@10 : 0.1781
  ```

- **Logging Result**:

  ```yaml
  78%|███████▊  | 126/162 [100:37:29<28:44:59, 2875.00s/trial, best loss: -0.2038]
  best params:  {'hidden_size_list': '[128,128,128]', 'learning_rate': 0.0005, 'message_dropout': 0.0, 'node_dropout': 0.0, 'reg_weight': 1e-05}
  best result: 
  {'model': 'NGCF', 'best_valid_score': 0.2038, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1639), ('mrr@10', 0.3661), ('ndcg@10', 0.2038), ('hit@10', 0.7104), ('precision@10', 0.1517)]), 'test_result': OrderedDict([('recall@10', 0.18), ('mrr@10', 0.4344), ('ndcg@10', 0.2494), ('hit@10', 0.7331), ('precision@10', 0.1845)])}
  ```
