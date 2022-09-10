# Knowledge-aware Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_kg.md)

- **Model**: [KGCN](https://recbole.io/docs/user_guide/model/knowledge/kgcn.html)

- **Time cost**: 7773.47s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.002,0.001,0.0005] 
  n_iter choice [1,2] 
  reg_weight choice [1e-3,1e-5,1e-7]
  aggregator choice ['sum','concat','neighbor']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  n_iter: 2
  reg_weight: 0.001
  aggregator: sum
  ```

- **Hyper-parameter logging** (hyper.result):

  | aggregator | reg_weight | learning_rate | n_iter | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |------------|------------|---------------|--------|------|----------------|-----------|--------|---------|
  | sum        | 0.001      | 0.001         | 2      | 1    | 15027.8        | 0.1373    | 0.064  | 0.0809  |
  | sum        | 1e-07      | 0.001         | 1      | 1    | 1439.98        | 0.0737    | 0.0377 | 0.0458  |
  | sum        | 0.001      | 0.001         | 2      | 1    | 15004.6        | 0.1373    | 0.064  | 0.0809  |
  | concat     | 1e-05      | 0.0005        | 2      | 1    | 2378.44        | 0.0459    | 0.023  | 0.0281  |
  | neighbor   | 0.001      | 0.002         | 2      | 1    | 5016.55        | 0.036     | 0.0137 | 0.0187  |


- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.001, 'n_iter': 2, 'aggregator': 'sum', 'reg_weight': 0.001}
  best result:  {'recall@10': 0.1373, 'mrr@10': 0.064, 'ndcg@10': 0.0809, 'hit@10': 0.1391, 'precision@10': 0.014, 'time_this_iter_s': 15027.762351036072}

  ```
