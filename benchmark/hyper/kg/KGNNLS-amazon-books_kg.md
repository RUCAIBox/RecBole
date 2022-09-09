# Knowledge-aware Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_kg.md)

- **Model**: [KGNNLS](https://recbole.io/docs/user_guide/model/knowledge/kgnnls.html)

- **Time cost**: 9741.83s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.002,0.001,0.0005] 
  n_iter choice [1,2] 
  reg_weight choice [1e-3,1e-5]
  ls_weight choice [1,0.5,0.1,0.01,0.001]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.002
  n_iter: 2
  reg_weight: 1e-3
  ls_weight: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  | reg_weight | learning_rate | ls_weight | n_iter | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |------------|---------------|-----------|--------|------|----------------|-----------|--------|---------|
  | 0.001      | 0.001         | 0.1       | 2      | 1    | 21384.9        | 0.1377    | 0.0663 | 0.0827  |
  | 0.001      | 0.001         | 0.01      | 1      | 1    | 1932.17        | 0.0737    | 0.0377 | 0.0458  |
  | 0.001      | 0.002         | 0.01      | 2      | 1    | 21198          | 0.1417    | 0.0719 | 0.088   |
  | 0.001      | 0.001         | 0.1       | 1      | 1    | 2030.91        | 0.0737    | 0.0377 | 0.0458  |
  | 1e-05      | 0.0005        | 0.1       | 1      | 1    | 2163.19        | 0.07      | 0.0344 | 0.0424  |



- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.002, 'n_iter': 2,'reg_weight': 0.001, 'ls_weight': 0.01}
  best result:  {'recall@10': 0.1417, 'mrr@10': 0.0719, 'ndcg@10': 0.088, 'hit@10': 0.1435, 'precision@10': 0.0144, 'time_this_iter_s': 21198.022414445877}

  ```
