# Knowledge-aware Recommendation

- **Dataset**: [Lastfm-track](../../md/lastfm-track_kg.md)

- **Model**: [CFKG](https://recbole.io/docs/user_guide/model/knowledge/cfkg.html)

- **Time cost**: 61732.64s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.01,0.005,0.0005]
  loss_function choice ['inner_product','transe']
  margin choice ['0.5,1.0,2.0']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  loss_function: inner_product
  margin: 1.0
  ```

- **Hyper-parameter logging** (hyper.result):

  | learning_rate | loss_function | margin | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |---------------|---------------|--------|------|----------------|-----------|--------|---------|
  | 0.0005        | inner_product | 1      | 1    | 98835.2        | 0.1549    | 0.1484 | 0.1169  |
  | 0.01          | transe        | 0.5    | 1    | 56839.3        | 0.0957    | 0.1211 | 0.0778  |
  | 0.0005        | inner_product | 1      | 1    | 98364.7        | 0.1549    | 0.1484 | 0.1169  |
  | 0.005         | transe        | 1      | 1    | 40323.1        | 0.1163    | 0.1429 | 0.0944  |
  | 0.005         | inner_product | 2      | 1    | 14300.9        | 0.1291    | 0.1231 | 0.0957  |

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0005, 'loss_function': 'inner_product', 'margin': 1.0}
  best result:  {'recall@10': 0.1549, 'mrr@10': 0.1484, 'ndcg@10': 0.1169, 'hit@10': 0.3287, 'precision@10': 0.0453, 'time_this_iter_s': 98835.23093175888}

  ```
