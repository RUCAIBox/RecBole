# Knowledge-aware Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_kg.md)

- **Model**: [KGAT](https://recbole.io/docs/user_guide/model/knowledge/kgat.html)

- **Time cost**: 2188.99s/trial

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

  | aggregator | reg_weight | learning_rate | n_iter | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |------------|------------|---------------|--------|------|----------------|-----------|--------|---------|
  | sum        | 0.001      | 0.001         | 2      | 1    | 15027.8        | 0.1373    | 0.064  | 0.0809  |
  | sum        | 1e-07      | 0.001         | 1      | 1    | 1439.98        | 0.0737    | 0.0377 | 0.0458  |
  | sum        | 0.001      | 0.001         | 2      | 1    | 15004.6        | 0.1373    | 0.064  | 0.0809  |
  | concat     | 1e-05      | 0.0005        | 2      | 1    | 2378.44        | 0.0459    | 0.023  | 0.0281  |
  | neighbor   | 0.001      | 0.002         | 2      | 1    | 5016.55        | 0.036     | 0.0137 | 0.0187  |



- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.001, 'n_iter': 2, 'aggregator': 'sum', 'l2_weight': 0.001, 'ls_weight': 1}
  best result:  {'recall@10': 0.1373, 'mrr@10': 0.064, 'ndcg@10': 0.0809, 'hit@10': 0.1391, 'precision@10': 0.014, 'time_this_iter_s': 15027.762351036072}

  ```
