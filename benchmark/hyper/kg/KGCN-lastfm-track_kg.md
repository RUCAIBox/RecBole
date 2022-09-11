# Knowledge-aware Recommendation

- **Dataset**: [Lastfm-track](../../md/lastfm-track_kg.md)

- **Model**: [KGCN](https://recbole.io/docs/user_guide/model/knowledge/kgcn.html)

- **Time cost**: 159294.4s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.002,0.001,0.0005] 
  n_iter choice [1,2] 
  reg_weight choice [1e-3,1e-5,1e-7]
  aggregator choice ['sum','concat','neighbor']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.002
  n_iter: 1
  reg_weight: 0.0001
  aggregator: sum
  ```

- **Hyper-parameter logging** (hyper.result):

  | aggregator | learning_rate | n_iter | reg_weight | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |------------|---------------|--------|------------|------|----------------|-----------|--------|---------|
  | sum        | 0.002         | 1      | 0.001      | 1    | 131830         | 0.1446    | 0.1388 | 0.1081  |
  | neighbor   | 0.001         | 1      | 1e-07      | 1    | 105915         | 0.1068    | 0.0985 | 0.0766  |
  | neighbor   | 0.0005        | 1      | 1e-05      | 1    | 296212         | 0.11      | 0.0979 | 0.0776  |
  | concat     | 0.002         | 2      | 1e-05      | 1    | 150037         | 0.1218    | 0.1082 | 0.0862  |
  | sum        | 0.002         | 1      | 1e-05      | 1    | 112478         | 0.1414    | 0.1338 | 0.1047  |


- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.002, 'n_iter': 1, 'aggregator': 'sum', 'reg_weight': 0.001}
  best result:  {'recall@10': 0.1446, 'mrr@10': 0.1388, 'ndcg@10': 0.1081, 'hit@10': 0.3153, 'precision@10': 0.0433, 'time_this_iter_s': 131829.8193845749}

  ```
