# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [NAIS](https://recbole.io/docs/user_guide/model/general/nais.html)

- **Time cost**: 1456.24s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [1e-4,1e-3,1e-2] 
  weight_size in [64,32] 
  reg_weights in ['[1e-7, 1e-7, 1e-5]', '[0, 0, 0]'] 
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 1e-4  
  reg_weights: [1e-7,1e-7,1e-5]     
  weight_size: 32
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001, reg_weights:'[0,0,0]', weight_size:32
  Valid result:
  recall@10 : 0.1676    mrr@10 : 0.3638    ndcg@10 : 0.2031    hit@10 : 0.7134    precision@10 : 0.1502
  Test result:
  recall@10 : 0.1857    mrr@10 : 0.4299    ndcg@10 : 0.2459    hit@10 : 0.7427    precision@10 : 0.181

  learning_rate:0.0001, reg_weights:'[1e-7,1e-7,1e-5]', weight_size:32
  Valid result:
  recall@10 : 0.1681    mrr@10 : 0.3631    ndcg@10 : 0.2033    hit@10 : 0.7136    precision@10 : 0.1504
  Test result:
  recall@10 : 0.1859    mrr@10 : 0.4297    ndcg@10 : 0.246     hit@10 : 0.7427    precision@10 : 0.181

  learning_rate:0.01, reg_weights:'[1e-7,1e-7,1e-5]', weight_size:64
  Valid result:
  recall@10 : 0.0924    mrr@10 : 0.2397    ndcg@10 : 0.1216    hit@10 : 0.5458    precision@10 : 0.0963
  Test result:
  recall@10 : 0.0966    mrr@10 : 0.2698    ndcg@10 : 0.1392    hit@10 : 0.5599    precision@10 : 0.111
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 12/12 [4:51:14<00:00, 1456.24s/trial, best loss: -0.2033]
  best params:  {'learning_rate': 0.0001, 'reg_weights': '[1e-7,1e-7,1e-5]', 'weight_size': 32}
  best result: 
  {'model': 'NAIS', 'best_valid_score': 0.2033, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1681), ('mrr@10', 0.3631), ('ndcg@10', 0.2033), ('hit@10', 0.7136), ('precision@10', 0.1504)]), 'test_result': OrderedDict([('recall@10', 0.1859), ('mrr@10', 0.4297), ('ndcg@10', 0.246), ('hit@10', 0.7427), ('precision@10', 0.181)])}
  ```
