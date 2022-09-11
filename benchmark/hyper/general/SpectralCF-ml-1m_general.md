# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [SLIMElastic](https://recbole.io/docs/user_guide/model/general/slimelastic.html)

- **Time cost**: 813.82s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.002,0.001,0.0005] 
  reg_weight choice [0.002,0.001,0.0005] 
  n_layers choice [1,2,3,4] 
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001  
  n_layers: 3  
  reg_weight: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.123     mrr@10 : 0.3185    ndcg@10 : 0.1662    hit@10 : 0.6259    precision@10 : 0.1272
  Test result:
  recall@10 : 0.1331    mrr@10 : 0.3617    ndcg@10 : 0.1974    hit@10 : 0.6421    precision@10 : 0.1534

  learning_rate:0.001, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.1431    mrr@10 : 0.3375    ndcg@10 : 0.1834    hit@10 : 0.6688    precision@10 : 0.1389
  Test result:
  recall@10 : 0.1569    mrr@10 : 0.3937    ndcg@10 : 0.2214    hit@10 : 0.6877    precision@10 : 0.1686

  learning_rate:0.0005, n_layers:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.143     mrr@10 : 0.3412    ndcg@10 : 0.1841    hit@10 : 0.6745    precision@10 : 0.1406
  Test result:
  recall@10 : 0.1571    mrr@10 : 0.3947    ndcg@10 : 0.2218    hit@10 : 0.6885    precision@10 : 0.1687
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [8:08:17<00:00, 813.82s/trial, best loss: -0.1842] 
  best params:  {'learning_rate': 0.001, 'n_layers': 3, 'reg_weight': 0.0005}
  best result: 
  {'model': 'SpectralCF', 'best_valid_score': 0.1842, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1419), ('mrr@10', 0.3372), ('ndcg@10', 0.1842), ('hit@10', 0.6662), ('precision@10', 0.1408)]), 'test_result': OrderedDict([('recall@10', 0.1554), ('mrr@10', 0.3989), ('ndcg@10', 0.2228), ('hit@10', 0.6871), ('precision@10', 0.1697)])}
  ```
