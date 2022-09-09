# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [GRU4RecF](https://recbole.io/docs/user_guide/model/sequential/gru4recf.html)

- **Time cost**: 1936.59s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  num_layers choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  num_layers: 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001, num_layers:2
  Valid result:
  recall@10 : 0.276    mrr@10 : 0.1156    ndcg@10 : 0.1529    hit@10 : 0.276    precision@10 : 0.0276
  Test result:
  recall@10 : 0.2599    mrr@10 : 0.1107    ndcg@10 : 0.1455    hit@10 : 0.2599    precision@10 : 0.026

  learning_rate:0.001, num_layers:1
  Valid result:
  recall@10 : 0.2846    mrr@10 : 0.1206    ndcg@10 : 0.1588    hit@10 : 0.2846    precision@10 : 0.0285
  Test result:
  recall@10 : 0.2573    mrr@10 : 0.1131    ndcg@10 : 0.1469    hit@10 : 0.2573    precision@10 : 0.0257

  learning_rate:0.0001, num_layers:2
  Valid result:
  recall@10 : 0.2503    mrr@10 : 0.101    ndcg@10 : 0.1358    hit@10 : 0.2503    precision@10 : 0.025
  Test result:
  recall@10 : 0.2478    mrr@10 : 0.1003    ndcg@10 : 0.1346    hit@10 : 0.2478    precision@10 : 0.0248

  learning_rate:0.0005, num_layers:2
  Valid result:
  recall@10 : 0.278    mrr@10 : 0.1116    ndcg@10 : 0.1504    hit@10 : 0.278    precision@10 : 0.0278
  Test result:
  recall@10 : 0.2621    mrr@10 : 0.106    ndcg@10 : 0.1423    hit@10 : 0.2621    precision@10 : 0.0262

  learning_rate:0.0005, num_layers:1
  Valid result:
  recall@10 : 0.2803    mrr@10 : 0.118    ndcg@10 : 0.1558    hit@10 : 0.2803    precision@10 : 0.028
  Test result:
  recall@10 : 0.2582    mrr@10 : 0.1103    ndcg@10 : 0.1448    hit@10 : 0.2582    precision@10 : 0.0258

  learning_rate:0.005, num_layers:2
  Valid result:
  recall@10 : 0.2896    mrr@10 : 0.1207    ndcg@10 : 0.16    hit@10 : 0.2896    precision@10 : 0.029
  Test result:
  recall@10 : 0.2609    mrr@10 : 0.1074    ndcg@10 : 0.1432    hit@10 : 0.2609    precision@10 : 0.0261

  learning_rate:0.005, num_layers:1
  Valid result:
  recall@10 : 0.2866    mrr@10 : 0.1212    ndcg@10 : 0.1597    hit@10 : 0.2866    precision@10 : 0.0287
  Test result:
  recall@10 : 0.2664    mrr@10 : 0.1169    ndcg@10 : 0.1517    hit@10 : 0.2664    precision@10 : 0.0266

  learning_rate:0.0001, num_layers:1
  Valid result:
  recall@10 : 0.2508    mrr@10 : 0.101    ndcg@10 : 0.1358    hit@10 : 0.2508    precision@10 : 0.0251
  Test result:
  recall@10 : 0.2415    mrr@10 : 0.0976    ndcg@10 : 0.1312    hit@10 : 0.2415    precision@10 : 0.0242
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 8/8 [4:18:12<00:00, 1936.59s/trial, best loss: -0.16]
  best params:  {'learning_rate': 0.005, 'num_layers': 2}
  best result: 
  {'model': 'GRU4RecF', 'best_valid_score': 0.16, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2896), ('mrr@10', 0.1207), ('ndcg@10', 0.16), ('hit@10', 0.2896), ('precision@10', 0.029)]), 'test_result': OrderedDict([('recall@10', 0.2609), ('mrr@10', 0.1074), ('ndcg@10', 0.1432), ('hit@10', 0.2609), ('precision@10', 0.0261)])}
  ```
