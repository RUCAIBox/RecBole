# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [MKR](https://recbole.io/docs/user_guide/model/knowledge/mkr.html)

- **Time cost**: 501.11s/trial

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
  learning_rate: 0.001
  low_layers_num: [1,2,3]
  high_layers_num: [1,2]
  l2_weight: [1e-6,1e-4]
  kg_embedding_size: [16,32,64]
  ```

- **Hyper-parameter logging** (hyper.result):

  | high_layers_num | kg_embedding_size | learning_rate | low_layers_num | reg_weight | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |-----------------|-------------------|---------------|----------------|------------|------|----------------|-----------|--------|---------|
  | 2               | 32                | 0.001         | 1              | 1e-06      | 1    | 5089.75        | 0.1584    | 0.3692 | 0.2115  |
  | 1               | 64                | 0.01          | 2              | 1e-06      | 1    | 3653.17        | 0.1303    | 0.3333 | 0.1842  |
  | 2               | 32                | 0.005         | 1              | 0.0001     | 1    | 1304.13        | 0.0525    | 0.1448 | 0.0693  |
  | 2               | 32                | 0.005         | 1              | 0.0001     | 1    | 2630.4         | 0.0792    | 0.2333 | 0.1197  |
  | 1               | 64                | 0.001         | 1              | 1e-06      | 1    | 4020.7         | 0.1573    | 0.374  | 0.2123  |


- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.001, 'loss_function': 'inner_product', 'margin': 1.0}
    best result:  {'recall@10': 0.1703, 'mrr@10': 0.4022, 'ndcg@10': 0.231, 'hit@10': 0.7139, 'precision@10': 0.1738, 'time_this_iter_s': 768.2000279426575}

  ```
