# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [DGCF](https://recbole.io/docs/user_guide/model/general/dgcf.html)

- **Time cost**: 10411.18s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-4,1e-3, 5e-3,1e-2]                     
  n_factors choice [2, 4, 8]                                                  
  reg_weight choice [1e-3,1e-2]                                         
  cor_weight choice [1e-3,1e-2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 5e-4  
  n_factors: 2  
  reg_weight: 0.001  
  cor_weight: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  cor_weight:0.001, learning_rate:0.005, n_factors:4, reg_weight:0.001
  Valid result:
  recall@10 : 0.1575    mrr@10 : 0.3529    ndcg@10 : 0.1964    hit@10 : 0.6952    precision@10 : 0.1462
  Test result:
  recall@10 : 0.1745    mrr@10 : 0.413     ndcg@10 : 0.2379    hit@10 : 0.7189    precision@10 : 0.178

  cor_weight:0.01, learning_rate:0.001, n_factors:4, reg_weight:0.01
  Valid result:
  recall@10 : 0.1655    mrr@10 : 0.3617    ndcg@10 : 0.2026    hit@10 : 0.7113    precision@10 : 0.1508
  Test result:
  recall@10 : 0.1848    mrr@10 : 0.4239    ndcg@10 : 0.2457    hit@10 : 0.7358    precision@10 : 0.1828

  cor_weight:0.001, learning_rate:0.0005, n_factors:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.1674    mrr@10 : 0.3639    ndcg@10 : 0.2046    hit@10 : 0.7152    precision@10 : 0.1523
  Test result:
  recall@10 : 0.1846    mrr@10 : 0.4287    ndcg@10 : 0.2477    hit@10 : 0.7376    precision@10 : 0.1838
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 48/48 [138:48:56<00:00, 10411.18s/trial, best loss: -0.2046]
  best params:  {'cor_weight': 0.001, 'learning_rate': 0.0005, 'n_factors': 2, 'reg_weight': 0.001}
  best result: 
  {'model': 'DGCF', 'best_valid_score': 0.2046, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1674), ('mrr@10', 0.3639), ('ndcg@10', 0.2046), ('hit@10', 0.7152), ('precision@10', 0.1523)]), 'test_result': OrderedDict([('recall@10', 0.1846), ('mrr@10', 0.4287), ('ndcg@10', 0.2477), ('hit@10', 0.7376), ('precision@10', 0.1838)])}
  ```
