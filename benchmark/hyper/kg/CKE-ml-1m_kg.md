# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [CKE](https://recbole.io/docs/user_guide/model/knowledge/cke.html)

- **Time cost**: 1427.49s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.001,0.0005,0.0007]
  kg_embedding_size choice [32，128]
  reg_weights choice [[0.1,0.1],[0.01,0.01],[0.001,0.001]]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  kg_embedding_size: 128
  reg_weights: [0.001,0.001]
  ```

- **Hyper-parameter logging** (hyper.result):

  | kg_embedding_size | learning_rate | reg_weights    | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |-------------------|---------------|----------------|------|----------------|-----------|--------|---------|
  | 128               | 0.0007        | [0.01, 0.01]   | 1    | 1080.2         | 0.1717    | 0.4073 | 0.2367  |
  | 128               | 0.0005        | [0.001, 0.001] | 1    | 2004.4         | 0.1768    | 0.4139 | 0.2405  |
  | 32                | 0.0007        | [0.01, 0.01]   | 1    | 1375.16        | 0.1747    | 0.4171 | 0.239   |
  | 32                | 0.001         | [0.1, 0.1]     | 1    | 1216.05        | 0.1725    | 0.4073 | 0.2347  |
  | 32                | 0.0005        | [0.01, 0.01]   | 1    | 1461.64        | 0.1739    | 0.415  | 0.2393  |


- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.0005, 'kg_embedding_size': 128, 'reg_weights': [0.001, 0.001]}
    best result:  {'recall@10': 0.1768, 'mrr@10': 0.4139, 'ndcg@10': 0.2405, 'hit@10': 0.7212, 'precision@10': 0.1803, 'time_this_iter_s': 2004.3991494178772}
  ```
