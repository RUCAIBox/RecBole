# Knowledge-aware Recommendation

- **Dataset**: [Lastfm-track](../../md/lastfm-track_kg.md)

- **Model**: [MKR](https://recbole.io/docs/user_guide/model/knowledge/mkr.html)

- **Time cost**: 98653.42s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [5e-5,1e-4,1e-3,5e-3,1e-2]
  low_layers_num in [1,2,3]
  high_layers_num in [1,2]
  l2_weight in [1e-6,1e-4]
  kg_embedding_size in [16,32,64]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  low_layers_num: 1
  high_layers_num: 2
  l2_weight: 1e-6
  kg_embedding_size: 32
  ```

- **Hyper-parameter logging** (hyper.result):

  | high_layers_num | kg_embedding_size | l2_weight | learning_rate | low_layers_num | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |-----------------|-------------------|-----------|---------------|----------------|------|----------------|-----------|--------|---------|
  | 2               | 16                | 1e-06     | 0.0001        | 3              | 1    | 48497.4        | 0.0107    | 0.0155 | 0.0083  |
  | 2               | 16                | 0.0001    | 0.001         | 2              | 1    | 348651         | 0.0972    | 0.0948 | 0.0717  |
  | 2               | 32                | 0.0001    | 0.0001        | 3              | 1    | 48840.5        | 0.0411    | 0.051  | 0.0324  |
  | 1               | 32                | 0.0001    | 0.01          | 3              | 1    | 3935.38        | 0.0278    | 0.0352 | 0.0209  |
  | 2               | 32                | 1e-6      | 0.005         | 1              | 1    | 43342.8        | 0.1036    | 0.102  | 0.0762  |



- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005, 'low_layers_num': 1, 'high_layers_num': 2, 'l2_weight': 1e-06, 'kg_embedding_size': 32}
  best result:  {'recall@10': 0.1036, 'mrr@10': 0.102, 'ndcg@10': 0.0762, 'hit@10': 0.2455, 'precision@10': 0.0318, 'time_this_iter_s': 43342.85371944904327}

  ```
