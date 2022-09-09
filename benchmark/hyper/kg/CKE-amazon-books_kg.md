# Knowledge-aware Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_kg.md)

- **Model**: [CKE](https://recbole.io/docs/user_guide/model/knowledge/cke.html)

- **Time cost**: 1427.49s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.001,0.0005,0.0007]
  kg_embedding_size choice [64，128]
  reg_weights choice [[0.1,0.1],[0.01,0.01],[0.001,0.001]]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0007
  kg_embedding_size: 128
  reg_weights: [0.001,0.001]
  ```

- **Hyper-parameter logging** (hyper.result):

  | kg_embedding_size | learning_rate | reg_weights    | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  | ----------------- | ------------- | -------------- | ---- | -------------- | --------- | ------ | ------- |
  | 128               | 0.0007        | [0.001, 0.001] | 1    | 1080.2         | 0.172     | 0.1933 | 0.1397  |
  | 128               | 0.0005        | [0.1, 0.1]     | 1    | 2004.4         | 0.0168    | 0.0288 | 0.0154  |
  | 64                | 0.0005        | [0.01, 0.01]   | 1    | 1375.16        | 0.0164    | 0.0283 | 0.015   |
  | 64                | 0.001         | [0.01, 0.01]   | 1    | 1216.05        | 0.1698    | 0.1868 | 0.136   |
  | 128               | 0.001         | [0.001, 0.001] | 1    | 1461.64        | 0.1689    | 0.1849 | 0.1347  |


- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.0007, 'kg_embedding_size': 128, 'reg_weights': [0.001, 0.001]}
    best result:  {'recall@10': 0.172, 'mrr@10': 0.1933, 'ndcg@10': 0.1397, 'hit@10': 0.3915, 'precision@10': 0.0589, 'time_this_iter_s': 24910.351511240005}
  ```
