# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [KGNNLS](https://recbole.io/docs/user_guide/model/knowledge/kgnnls.html)

- **Time cost**: 6488.48s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.002,0.001,0.0005] 
  n_iter choice [1,2] 
  reg_weight choice [1e-3,1e-5]
  ls_weight choice [1,0.5,0.1,0.01,0.001]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  n_iter: 1
  reg_weight: 1e-5
  ls_weight: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  | reg_weight | learning_rate | ls_weight | n_iter | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |------------|---------------|-----------|--------|------|----------------|-----------|--------|---------|
  | 1e-05      | 0.0005        | 1         | 1      | 1    | 12643          | 0.1587    | 0.3826 | 0.2175  |
  | 1e-05      | 0.002         | 0.01      | 2      | 1    | 5495.43        | 0.1472    | 0.3725 | 0.206   |
  | 0.001      | 0.0005        | 0.001     | 1      | 1    | 12772          | 0.1576    | 0.3804 | 0.2165  |
  | 1e-05      | 0.002         | 0.01      | 2      | 1    | 998.999        | 0.1506    | 0.3746 | 0.2086  |
  | 1e-05      | 0.002         | 0.01      | 1      | 1    | 533.001        | 0.1506    | 0.3763 | 0.2116  |


- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0005, 'n_iter': 1, 'aggregator': 'sum', 'l2_weight': 1e-05, 'ls_weight': 1, 'neighbor_sample_size': 4}
  best result:  {'recall@10': 0.1587, 'mrr@10': 0.3826, 'ndcg@10': 0.2175, 'hit@10': 0.6857, 'precision@10': 0.1659, 'time_this_iter_s': 12642.982091665268}

  ```
