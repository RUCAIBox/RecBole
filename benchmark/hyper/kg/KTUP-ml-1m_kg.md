# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [KTUP](https://recbole.io/docs/user_guide/model/knowledge/ktup.html)

- **Time cost**: 1515.92s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.01,0.005,0.001]
  use_st_gumbel choice [True, False]
  train_rec_step choice [8,10]
  train_kg_step choice [0,1,2,3,4,5]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  use_st_gumbel: True
  train_rec_step: 8
  train_kg_step: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  | learning_rate | train_kt_step | train_rec_step | use_st_gumbel | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |   |
  |---------------|---------------|----------------|---------------|------|----------------|-----------|--------|---------|---|
  | 0.005         | 4             | 8              | False         | 1    | 684.657        | 0.121     | 0.3057 | 0.1694  |   |
  | 0.01          | 0             | 10             | False         | 1    | 1404.92        | 0.1106    | 0.2851 | 0.1541  |   |
  | 0.01          | 3             | 10             | True          | 1    | 1154.17        | 0.1145    | 0.2996 | 0.1593  |   |
  | 0.001         | 1             | 8              | True          | 1    | 2444.14        | 0.1463    | 0.3457 | 0.1963  |   |
  | 0.001         | 3             | 10             | False         | 1    | 1891.74        | 0.1343    | 0.3323 | 0.1881  |   |



- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.001, 'use_st_gumbel': True, 'train_rec_step': 8, 'train_kg_step': 1}
    best result:  {'recall@10': 0.1463, 'mrr@10': 0.3457, 'ndcg@10': 0.1963, 'hit@10': 0.6582, 'precision@10': 0.1524, 'time_this_iter_s': 2444.1433942317963}

  ```
