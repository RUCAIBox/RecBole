# Knowledge-aware Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_kg.md)

- **Model**: [KGAT](https://recbole.io/docs/user_guide/model/knowledge/kgat.html)

- **Time cost**: 7773.47s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0001]
  layers choice [[64,32,16],[64,64,64],[128,64,32]]
  reg_weight choice [5e-5,1e-5,5e-6,1e-6]
  mess_dropout choice [0.1,0.2,0.5]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  layers: [64, 32, 16]
  reg_weight: 5e-5
  mess_dropout: 0.5
  ```

- **Hyper-parameter logging** (hyper.result):

  | layers       | learning_rate | mess_dropout | reg_weight | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |--------------|---------------|--------------|------------|------|----------------|-----------|--------|---------|
  | [64, 64, 64] | 0.0001        | 0.3          | 1e-05      | 1    | 1410.74        | 0.0602    | 0.0285 | 0.0356  |
  | [64, 32, 16] | 0.0001        | 0.2          | 5e-05      | 1    | 1549.42        | 0.0602    | 0.0285 | 0.0356  |
  | [64, 32, 16] | 0.01          | 0.4          | 5e-06      | 1    | 1841.19        | 0.1353    | 0.0726 | 0.0869  |
  | [64, 64, 64] | 0.0005        | 0.4          | 0.0001     | 1    | 1007.35        | 0.07      | 0.0345 | 0.0425  |
  | [64, 32, 16] | 0.005         | 0.5          | 5e-05      | 1    | 2628.36        | 0.1528    | 0.0842 | 0.1     |


- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005, 'layers': [64, 32, 16], 'reg_weight': 5e-05, 'mess_dropout': 0.5}
  best result:  {'recall@10': 0.1528, 'mrr@10': 0.0842, 'ndcg@10': 0.1, 'hit@10': 0.1551, 'precision@10': 0.0156, 'time_this_iter_s': 2628.3564019203186}

  ```
