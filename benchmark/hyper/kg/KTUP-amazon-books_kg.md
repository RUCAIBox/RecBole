# Knowledge-aware Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_kg.md)

- **Model**: [KTUP](https://recbole.io/docs/user_guide/model/knowledge/ktup.html)

- **Time cost**: 7328.08s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.01,0.005,0.001]
  L1_flag choice [True, False]
  use_st_gumbel choice [True, False]
  train_rec_step choice [8,10]
  train_kg_step choice [0,1,2,3,4,5]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  L1_flag: False
  use_st_gumbel: True
  train_rec_step: 8
  train_kg_step: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  | L1_flag | learning_rate | train_kg_step | train_rec_step | use_st_gumbel | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  | ------- | ------------- | ------------- | -------------- | ------------- | ---- | -------------- | --------- | ------ | ------- |
  | False   | 0.0001        | 3             | 8              | True          | 1    | 8640.9         | 0.0903    | 0.0407 | 0.052   |
  | True    | 0.005         | 4             | 8              | False         | 1    | 2731.36        | 0.0896    | 0.0505 | 0.0593  |
  | False   | 0.0001        | 4             | 8              | True          | 1    | 8607.58        | 0.0883    | 0.0395 | 0.0506  |
  | False   | 0.005         | 1             | 8              | True          | 1    | 7840.33        | 0.1959    | 0.1152 | 0.1336  |
  | True    | 0.01          | 1             | 10             | True          | 1    | 8820.27        | 0.1158    | 0.0649 | 0.0764  |



- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005, 'L1_flag': False, 'use_st_gumbel': True, 'train_rec_step': 8, 'train_kg_step': 1}
  best result:  {'recall@10': 0.1959, 'mrr@10': 0.1152, 'ndcg@10': 0.1336, 'hit@10': 0.1986, 'precision@10': 0.02, 'time_this_iter_s': 7840.33078619957}
  ```
