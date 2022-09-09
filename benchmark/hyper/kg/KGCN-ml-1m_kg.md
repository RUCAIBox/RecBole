# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [KGCN](https://recbole.io/docs/user_guide/model/knowledge/kgcn.html)

- **Time cost**: 2849.81s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.002,0.001,0.0005] 
  n_iter choice [1,2] 
  reg_weight choice [1e-4,5e-5,1e-5,5e-6,1e-6]
  aggregator choice ['sum','concat','neighbor']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.002
  n_iter: 2
  reg_weight: 1e-5 
  aggregator: sum
  ```

- **Hyper-parameter logging** (hyper.result):

  | aggregator | l2_weight | learning_rate | n_iter | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  | ---------- | --------- | ------------- | ------ | ---- | -------------- | --------- | ------ | ------- |
  | neighbor   | 1e-05     | 0.001         | 2      | 1    | 4011.85        | 0.112     | 0.3132 | 0.1642  |
  | neighbor   | 1e-07     | 0.002         | 1      | 1    | 2281.55        | 0.1332    | 0.3452 | 0.1873  |
  | sum        | 1e-07     | 0.0005        | 2      | 1    | 4063.48        | 0.1405    | 0.3579 | 0.1976  |
  | neighbor   | 1e-07     | 0.002         | 2      | 1    | 2686.34        | 0.1133    | 0.3137 | 0.1654  |
  | sum        | 1e-05     | 0.002         | 2      | 1    | 1205.85        | 0.1506    | 0.3746 | 0.2086  |


- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.002, 'n_iter': 2, 'aggregator': 'sum', 'reg_weight': 1e-05}
    best result:  {'recall@10': 0.1506, 'mrr@10': 0.3746, 'ndcg@10': 0.2086, 'hit@10': 0.6725, 'precision@10': 0.1571, 'time_this_iter_s': 1205.854009628296}

  ```
