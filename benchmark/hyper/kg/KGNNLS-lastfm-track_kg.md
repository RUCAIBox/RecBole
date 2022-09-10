# Knowledge-aware Recommendation

- **Dataset**: [Lastfm-track](../../md/lastfm-track_kg.md)
  
- **Model**: [KGNNLS](https://recbole.io/docs/user_guide/model/knowledge/kgnnls.html)

- **Time cost**: 156396.6s/trial

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
  n_iter: 1
  reg_weight: 1e-5
  ls_weight: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  | reg_weight | learning_rate | ls_weight | n_iter | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |------------|---------------|-----------|--------|------|----------------|-----------|--------|---------|
  | 0.001      | 0.001         | 1         | 1      | 1    | 172280         | 0.1307    | 0.1205 | 0.0951  |
  | 0.001      | 0.0005        | 0.5       | 1      | 1    | 187472         | 0.1146    | 0.1039 | 0.0815  |
  | 0.00001    | 0.002         | 0.001     | 1      | 1    | 90295.9        | 0.1369    | 0.1309 | 0.1017  |
  | 0.001      | 0.002         | 0.5       | 2      | 1    | 169072         | 0.1217    | 0.1104 | 0.0875  |
  | 0.001      | 0.002         | 0.5       | 2      | 1    | 162864         | 0.1142    | 0.1104 | 0.082   |




- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.002, 'n_iter': 2,'reg_weight': 0.00001, 'ls_weight': 0.001}
  best result:  {"recall@10": 0.1369, "mrr@10": 0.1309, "ndcg@10": 0.1017, "hit@10": 0.2992, "precision@10": 0.0401, "time_this_iter_s": 90295.8753566742}

  ```
