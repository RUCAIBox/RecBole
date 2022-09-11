# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [RippleNet](https://recbole.io/docs/user_guide/model/knowledge/cfkg.html)

- **Time cost**: 18476.26s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.001,0.005,0.01,0.05]
  n_memory choice [4,8,16,32]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  n_memory: 32
  ```

- **Hyper-parameter logging** (hyper.result):

  | learning_rate | n_memory | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |---------------|----------|------|----------------|-----------|--------|---------|
  | 0.005         | 16       | 1    | 17250.9        | 0.1177    | 0.1291 | 0.0894  |
  | 0.005         | 32       | 1    | 16961.6        | 0.1476    | 0.1579 | 0.1122  |
  | 0.01          | 4        | 1    | 12394.3        | 0.0108    | 0.0172 | 0.0091  |
  | 0.001         | 4        | 1    | 37257.4        | 0.0586    | 0.0661 | 0.0445  |
  | 0.005         | 4        | 1    | 8517.1         | 0.0347    | 0.0439 | 0.0267  |


- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005, 'n_memory': 32}
  best result:  {'recall@10': 0.1476, 'mrr@10': 0.1579, 'ndcg@10': 0.1122, 'hit@10': 0.3675, 'precision@10': 0.0544, 'time_this_iter_s': 16961.59213399887}
  ```
