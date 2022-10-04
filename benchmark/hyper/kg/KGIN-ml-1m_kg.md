# Knowledge-aware Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_kg.md)

- **Model**: [KGIN](https://recbole.io/docs/user_guide/model/knowledge/kgin.html)

- **Time cost**: 7564.57s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,1e-3,5e-3]
  node_dropout_rate choice [0.1,0.3,0.5]
  mess_dropout_rate choice [0.0,0.1]
  context_hops choice [2,3]
  n_factors choice [4,8]
  ind choice ['cosine','distance']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 1e-4
  node_dropout_rate: 0.5
  mess_dropout_rate: 0.0
  context_hops: 2
  n_factors: 4
  ind: 'cosine'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  context_hops:2, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1657    mrr@10 : 0.3624    ndcg@10 : 0.2021    hit@10 : 0.7195    precision@10 : 0.1508
  Test result:
  recall@10 : 0.1826    mrr@10 : 0.419    ndcg@10 : 0.2417    hit@10 : 0.7285    precision@10 : 0.1787
  
  context_hops:3, ind:cosine, learning_rate:0.005, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.1389    mrr@10 : 0.3268    ndcg@10 : 0.1775    hit@10 : 0.6667    precision@10 : 0.1352
  Test result:
  recall@10 : 0.1518    mrr@10 : 0.3648    ndcg@10 : 0.2032    hit@10 : 0.6822    precision@10 : 0.154
  
  context_hops:2, ind:cosine, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1664    mrr@10 : 0.3698    ndcg@10 : 0.2062    hit@10 : 0.7185    precision@10 : 0.1534
  Test result:
  recall@10 : 0.1895    mrr@10 : 0.4311    ndcg@10 : 0.2535    hit@10 : 0.7384    precision@10 : 0.1878
  
  context_hops:2, ind:cosine, learning_rate:0.0001, mess_dropout_rate:0.0, n_factors:4, node_dropout_rate:0.5
  Valid result:
  recall@10 : 0.1679    mrr@10 : 0.3718    ndcg@10 : 0.2073    hit@10 : 0.7219    precision@10 : 0.1541
  Test result:
  recall@10 : 0.1895    mrr@10 : 0.4331    ndcg@10 : 0.2529    hit@10 : 0.7424    precision@10 : 0.1868
  
  context_hops:2, ind:distance, learning_rate:0.005, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1504    mrr@10 : 0.3397    ndcg@10 : 0.1859    hit@10 : 0.6887    precision@10 : 0.1392
  Test result:
  recall@10 : 0.1582    mrr@10 : 0.3622    ndcg@10 : 0.2039    hit@10 : 0.688    precision@10 : 0.154
  
  context_hops:3, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.1, n_factors:8, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.159    mrr@10 : 0.3515    ndcg@10 : 0.1965    hit@10 : 0.7026    precision@10 : 0.1479
  Test result:
  recall@10 : 0.1788    mrr@10 : 0.4111    ndcg@10 : 0.2368    hit@10 : 0.7288    precision@10 : 0.1767
  
  context_hops:3, ind:cosine, learning_rate:0.001, mess_dropout_rate:0.0, n_factors:4, node_dropout_rate:0.1
  Valid result:
  recall@10 : 0.159    mrr@10 : 0.3548    ndcg@10 : 0.1976    hit@10 : 0.7089    precision@10 : 0.1483
  Test result:
  recall@10 : 0.176    mrr@10 : 0.4087    ndcg@10 : 0.2347    hit@10 : 0.7219    precision@10 : 0.1748
  
  context_hops:2, ind:cosine, learning_rate:0.005, mess_dropout_rate:0.0, n_factors:4, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1478    mrr@10 : 0.3351    ndcg@10 : 0.1841    hit@10 : 0.6867    precision@10 : 0.14
  Test result:
  recall@10 : 0.153    mrr@10 : 0.3566    ndcg@10 : 0.1984    hit@10 : 0.6827    precision@10 : 0.1497
  
  context_hops:2, ind:cosine, learning_rate:0.0001, mess_dropout_rate:0.1, n_factors:4, node_dropout_rate:0.3
  Valid result:
  recall@10 : 0.1593    mrr@10 : 0.3583    ndcg@10 : 0.1979    hit@10 : 0.7051    precision@10 : 0.1479
  Test result:
  recall@10 : 0.1781    mrr@10 : 0.4216    ndcg@10 : 0.2429    hit@10 : 0.7202    precision@10 : 0.1801
  ```


- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  82%|████████▏ | 118/144 [247:56:59<54:37:58, 7564.57s/trial, best loss: -0.2073]
  best params:  {'context_hops': 2, 'ind': 'cosine', 'learning_rate': 0.0001, 'mess_dropout_rate': 0.0, 'n_factors': 4, 'node_dropout_rate': 0.5}
  best result: 
  {'model': 'KGIN', 'best_valid_score': 0.2073, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1679), ('mrr@10', 0.3718), ('ndcg@10', 0.2073), ('hit@10', 0.7219), ('precision@10', 0.1541)]), 'test_result': OrderedDict([('recall@10', 0.1895), ('mrr@10', 0.4331), ('ndcg@10', 0.2529), ('hit@10', 0.7424), ('precision@10', 0.1868)])}
  ```
