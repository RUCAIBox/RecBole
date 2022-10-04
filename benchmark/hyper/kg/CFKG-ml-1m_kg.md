# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [CFKG](https://recbole.io/docs/user_guide/model/knowledge/cfkg.html)

- **Time cost**: 501.11s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.01,0.005,0.001,0.0001]
  loss_function choice ['inner_product','transe']
  margin choice ['0.5,1.0,2.0']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  loss_function: inner_product
  margin: 1.0
  ```

- **Hyper-parameter logging** (hyper.result):

  | learning_rate | loss_function | margin | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |---------------|---------------|--------|------|----------------|-----------|--------|---------|
  | 0.001         | inner_product | 1      | 1    | 768.2          | 0.1703    | 0.4022 | 0.231   |
  | 0.0001        | transe        | 2      | 1    | 956.75         | 0.0993    | 0.2809 | 0.1465  |
  | 0.005         | inner_product | 1      | 1    | 255.197        | 0.1582    | 0.3854 | 0.2186  |
  | 0.0001        | inner_product | 0.5    | 1    | 286.943        | 0.0782    | 0.2325 | 0.1195  |
  | 0.0001        | inner_product | 2      | 1    | 238.468        | 0.0782    | 0.2325 | 0.1195  |

- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.001, 'loss_function': 'inner_product', 'margin': 1.0}
    best result:  {'recall@10': 0.1703, 'mrr@10': 0.4022, 'ndcg@10': 0.231, 'hit@10': 0.7139, 'precision@10': 0.1738, 'time_this_iter_s': 768.2000279426575}

  ```
