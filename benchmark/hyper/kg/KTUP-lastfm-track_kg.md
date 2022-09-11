# Knowledge-aware Recommendation

- **Dataset**: [Lastfm-track](../../md/lastfm-track_kg.md)

- **Model**: [KTUP](https://recbole.io/docs/user_guide/model/knowledge/ktup.html)

- **Time cost**: 65757.63s/trial

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
  learning_rate: 0.0005
  L1_flag: True
  use_st_gumbel: False
  train_rec_step: 8
  train_kg_step: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  | L1_flag | learning_rate | train_kt_step | train_rec_step | use_st_gumbel | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |---------|---------------|---------------|----------------|---------------|------|----------------|-----------|--------|---------|
  | True    | 0.0005        | 1             | 8              | False         | 1    | 96133.4        | 0.1526    | 0.1489 | 0.1161  |
  | False   | 0.01          | 1             | 10             | False         | 1    | 22593          | 0.1066    | 0.1023 | 0.0778  |
  | False   | 0.0005        | 0             | 8              | False         | 1    | 97399.3        | 0.1526    | 0.1489 | 0.1161  |
  | False   | 0.0005        | 0             | 8              | True          | 1    | 93754.9        | 0.1526    | 0.1489 | 0.1161  |
  | True    | 0.01          | 1             | 8              | True          | 1    | 18907.6        | 0.1066    | 0.1023 | 0.0778  |



- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0005, 'L1_flag': True, 'use_st_gumbel': False, 'train_rec_step': 8, 'train_kt_step': 1}
  best result:  {'recall@10': 0.1526, 'mrr@10': 0.1489, 'ndcg@10': 0.1161, 'hit@10': 0.327, 'precision@10': 0.045, 'time_this_iter_s': 96133.36078619957}
  ```
