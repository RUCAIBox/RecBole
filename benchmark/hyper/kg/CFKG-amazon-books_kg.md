# Knowledge-aware Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_kg.md)

- **Model**: [CFKG](https://recbole.io/docs/user_guide/model/knowledge/cfkg.html)

- **Time cost**: 4561.77s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.01,0.005,0.001,0.0001]
  loss_function choice ['inner_product','transe']
  margin choice ['0.5,1.0,2.0']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  loss_function: transe
  margin: 0.5
  ```

- **Hyper-parameter logging** (hyper.result):

  | learning_rate | loss_function | margin | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |---------------|---------------|--------|------|----------------|-----------|--------|---------|
  | 0.001         | transe        | 0.5    | 1    | 6282.53        | 0.2231    | 0.1235 | 0.1462  |
  | 0.0005        | inner_product | 0.5    | 1    | 943.368        | 0.0658    | 0.0268 | 0.0356  |
  | 0.0001        | transe        | 2      | 1    | 8705.68        | 0.1434    | 0.0849 | 0.0982  |
  | 0.01          | transe        | 2      | 1    | 3874.81        | 0.2192    | 0.1186 | 0.1416  |
  | 0.005         | transe        | 1      | 1    | 3001.8         | 0.215     | 0.1174 | 0.1396  |


- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.001, 'loss_function': 'transe', 'margin': 0.5}
    best result:  {'recall@10': 0.2231, 'mrr@10': 0.1235, 'ndcg@10': 0.1462, 'hit@10': 0.2264, 'precision@10': 0.0229, 'time_this_iter_s': 6282.531406879425}

  ```
