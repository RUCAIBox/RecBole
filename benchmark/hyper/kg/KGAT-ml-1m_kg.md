# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [KGAT](https://recbole.io/docs/user_guide/model/knowledge/kgat.html)

- **Time cost**: 2330.26s/trial

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
  layers: [128, 64, 32]
  reg_weight: 5e-5
  mess_dropout: 0.1
  ```

- **Hyper-parameter logging** (hyper.result):

  | layers        | learning_rate | mess_dropout | reg_weight | iter | total time (s) | recall@10 | mrr@10 | ndcg@10 |
  |---------------|---------------|--------------|------------|------|----------------|-----------|--------|---------|
  | [64, 64, 64]  | 0.005         | 0.2          | 5e-06      | 1    | 764.223        | 0.15      | 0.3668 | 0.2062  |
  | [128, 64, 32] | 0.001         | 0.1          | 5e-05      | 1    | 2956.47        | 0.1783    | 0.4125 | 0.2404  |
  | [64, 64, 64]  | 0.001         | 0.2          | 1e-06      | 1    | 4060.2         | 0.1774    | 0.412  | 0.2372  |
  | [64, 32, 16]  | 0.0001        | 0.5          | 5e-06      | 1    | 1546.8         | 0.0775    | 0.2277 | 0.1139  |
  | [128, 64, 32] | 0.0001        | 0.5          | 1e-05      | 1    | 2323.63        | 0.0664    | 0.1973 | 0.097   |


- **Logging Result**:

  ```yaml
    best params:  {'learning_rate': 0.001, 'mess_dropout': 0.1, 'reg_weight': 5e-05, 'layers': [128, 64, 32], 'kg_embedding_size': 16, 'reg_weights': [0.1, 0.1]}
    best result:  {'recall@10': 0.1783, 'mrr@10': 0.4125, 'ndcg@10': 0.2404, 'hit@10': 0.722, 'precision@10': 0.1809, 'time_this_iter_s': 2956.4689252376556}

  ```
