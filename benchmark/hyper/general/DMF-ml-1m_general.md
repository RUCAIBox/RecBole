# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [DMF](https://recbole.io/docs/user_guide/model/general/dmf.html)

- **Time cost**: 4015.66s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,5e-4,3e-4,1e-4,5e-3,1e-3] 
  user_hidden_size_list choice ['[64, 64]','[64, 32]'] 
  item_hidden_size_list choice ['[64, 64]','[64, 32]']
  ```

- **Best parameters**:

  ```yaml
   learning_rate: 5e-4                                         
   user_hidden_size_list: '[64,64]'                        
   item_hidden_size_list: '[64,64]'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  item_hidden_size_list:'[64,32]', learning_rate:0.0003, user_hidden_size_list:'[64,32]'
  Valid result:
  recall@10 : 0.1445    mrr@10 : 0.338     ndcg@10 : 0.1842    hit@10 : 0.6705    precision@10 : 0.1394
  Test result:
  recall@10 : 0.1575    mrr@10 : 0.3969    ndcg@10 : 0.2223    hit@10 : 0.6884    precision@10 : 0.1673

  item_hidden_size_list:'[64,64]', learning_rate:0.0003, user_hidden_size_list:'[64,64]'
  Valid result:
  recall@10 : 0.1542    mrr@10 : 0.3482    ndcg@10 : 0.1917    hit@10 : 0.6864    precision@10 : 0.1433
  Test result:
  recall@10 : 0.1651    mrr@10 : 0.4018    ndcg@10 : 0.2263    hit@10 : 0.7069    precision@10 : 0.1692

  item_hidden_size_list:'[64,64]', learning_rate:0.0005, user_hidden_size_list:'[64,64]'
  Valid result:
  recall@10 : 0.1556    mrr@10 : 0.3499    ndcg@10 : 0.1926    hit@10 : 0.692     precision@10 : 0.1434
  Test result:
  recall@10 : 0.1698    mrr@10 : 0.4105    ndcg@10 : 0.2315    hit@10 : 0.7147    precision@10 : 0.1716
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 6/6 [6:41:33<00:00, 4015.66s/trial, best loss: -0.1926]
  best params:  {'item_hidden_size_list': '[64,64]', 'learning_rate': 0.0005, 'user_hidden_size_list': '[64,64]'}
  best result: 
  {'model': 'DMF', 'best_valid_score': 0.1926, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1556), ('mrr@10', 0.3499), ('ndcg@10', 0.1926), ('hit@10', 0.692), ('precision@10', 0.1434)]), 'test_result': OrderedDict([('recall@10', 0.1698), ('mrr@10', 0.4105), ('ndcg@10', 0.2315), ('hit@10', 0.7147), ('precision@10', 0.1716)])}
  ```
