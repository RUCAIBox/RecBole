# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [RippleNet](https://recbole.io/docs/user_guide/model/knowledge/cfkg.html)

- **Time cost**: 501.11s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.001,0.005,0.01,0.05]
  n_memory choice [4,8,16,32]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  n_memory: 32
  ```

- **Hyper-parameter logging** (hyper.result):

  | learning_rate | n_memory | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |---------------|----------|------|----------------|-----------|--------|---------|
  | 0.001         | 32       | 1    | 4681.92        | 0.1279    | 0.3335 | 0.1831  |
  | 0.01          | 8        | 1    | 1702.9         | 0.0793    | 0.2305 | 0.1191  |
  | 0.001         | 4        | 1    | 3061.87        | 0.1098    | 0.3026 | 0.1617  |
  | 0.001         | 16       | 1    | 3386.41        | 0.1244    | 0.3336 | 0.1818  |
  | 0.01          | 8        | 1    | 1865.72        | 0.0793    | 0.2305 | 0.1191  |


- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.001, 'n_memory': 32}
    best result:  {'recall@10': 0.1279, 'mrr@10': 0.3335, 'ndcg@10': 0.1831, 'hit@10': 0.6287, 'precision@10': 0.1431, 'time_this_iter_s': 4681.924859285355}

  ```
