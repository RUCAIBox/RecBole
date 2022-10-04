# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [RippleNet](https://recbole.io/docs/user_guide/model/knowledge/cfkg.html)

- **Time cost**: 1834.38s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.001,0.005,0.01,0.05]
  n_memory choice [4,8,16,32]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  n_memory: 4
  ```

- **Hyper-parameter logging** (hyper.result):

  | learning_rate | n_memory | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |---------------|----------|------|----------------|-----------|--------|---------|
  | 0.05          | 4        | 1    | 1485.74        | 0.0279    | 0.0095 | 0.0136  |
  | 0.05          | 4        | 1    | 1657.56        | 0.0279    | 0.0095 | 0.0136  |
  | 0.01          | 32       | 1    | 2442.84        | 0.1358    | 0.0723 | 0.0869  |
  | 0.05          | 16       | 1    | 1413.98        | 0.0213    | 0.0072 | 0.0104  |
  | 0.005         | 4        | 1    | 2171.8         | 0.154     | 0.0803 | 0.0973  |


- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005, 'n_memory': 4}
  best result:  {'recall@10': 0.154, 'mrr@10': 0.0803, 'ndcg@10': 0.0973, 'hit@10': 0.1562, 'precision@10': 0.0157, 'time_this_iter_s': 2171.801709651947}

  ```
