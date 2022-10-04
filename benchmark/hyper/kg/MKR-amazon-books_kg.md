# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [MKR](https://recbole.io/docs/user_guide/model/knowledge/mkr.html)

- **Time cost**: 2966.79s/trial

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
  low_layers_num: 2
  high_layers_num: 1
  l2_weight: 1e-6
  kg_embedding_size: 16
  ```

- **Hyper-parameter logging** (hyper.result):

  | high_layers_num | kg_embedding_size | l2_weight | learning_rate | low_layers_num | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |-----------------|-------------------|-----------|---------------|----------------|------|----------------|-----------|--------|---------|
  | 2               | 16                | 0.0001    | 0.001         | 1              | 1    | 3347.94        | 0.0652    | 0.0266 | 0.0353  |
  | 2               | 64                | 1e-06     | 0.005         | 1              | 1    | 2808.51        | 0.0626    | 0.0252 | 0.0336  |
  | 2               | 64                | 0.0001    | 0.01          | 2              | 1    | 3420.55        | 0.062     | 0.024  | 0.0326  |
  | 1               | 16                | 1e-06     | 0.005         | 2              | 1    | 3540.54        | 0.1882    | 0.1109 | 0.1288  |
  | 2               | 32                | 1e-06     | 0.0001        | 1              | 1    | 1716.45        | 0.066     | 0.0267 | 0.0356  |


- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005, 'low_layers_num': 2, 'high_layers_num': 1, 'l2_weight': 1e-06, 'kg_embedding_size': 16}
  best result:  {'recall@10': 0.1882, 'mrr@10': 0.1109, 'ndcg@10': 0.1288, 'hit@10': 0.1908, 'precision@10': 0.0192, 'time_this_iter_s': 3540.5371944904327}

  ```
