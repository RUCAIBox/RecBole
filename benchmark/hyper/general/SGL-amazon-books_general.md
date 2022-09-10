# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [SGL](https://recbole.io/docs/user_guide/model/general/sgl.html)

- **Time cost**: 8702.26s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml                                                            
  ssl_tau choice [0.1,0.2,0.5] 
  drop_ratio choice [0.1,0.2,0.4,0.5] 
  ssl_weight choice [0.05,0.1,0.5]
  ```

- **Best parameters**:

  ```yaml
  ssl_tau: 0.2  
  drop_ratio: 0.2  
  ssl_weight: 0.1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  drop_ratio:0.2, ssl_tau:0.2, ssl_weight:0.1
  Valid result:
  recall@10 : 0.1944    mrr@10 : 0.1425    ndcg@10 : 0.1296    hit@10 : 0.3068    precision@10 : 0.0387
  Test result:
  recall@10 : 0.1999    mrr@10 : 0.1558    ndcg@10 : 0.1388    hit@10 : 0.3131    precision@10 : 0.0408
  
  drop_ratio:0.5, ssl_tau:0.5, ssl_weight:0.5
  Valid result:
  recall@10 : 0.1697    mrr@10 : 0.1228    ndcg@10 : 0.1113    hit@10 : 0.2751    precision@10 : 0.034
  Test result:
  recall@10 : 0.1748    mrr@10 : 0.1318    ndcg@10 : 0.1185    hit@10 : 0.279     precision@10 : 0.0358

  drop_ratio:0.5, ssl_tau:0.1, ssl_weight:0.5
  Valid result:
  recall@10 : 0.1389    mrr@10 : 0.101     ndcg@10 : 0.0911    hit@10 : 0.2269    precision@10 : 0.0269
  Test result:
  recall@10 : 0.1416    mrr@10 : 0.1072    ndcg@10 : 0.0955    hit@10 : 0.2311    precision@10 : 0.0279
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [87:01:21<00:00, 8702.26s/trial, best loss: -0.1296]
  best params:  {'drop_ratio': 0.2, 'ssl_tau': 0.2, 'ssl_weight': 0.1, 'type': 'ED'}
  best result: 
  {'model': 'SGL', 'best_valid_score': 0.1296, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1944), ('mrr@10', 0.1425), ('ndcg@10', 0.1296), ('hit@10', 0.3068), ('precision@10', 0.0387)]), 'test_result': OrderedDict([('recall@10', 0.1999), ('mrr@10', 0.1558), ('ndcg@10', 0.1388), ('hit@10', 0.3131), ('precision@10', 0.0408)])}
  ```
