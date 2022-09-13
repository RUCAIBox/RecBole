# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [RecVAE](https://recbole.io/docs/user_guide/model/general/recvae.html)

- **Time cost**: 1038.60s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,7e-4,1e-3,5e-3,7e-3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0007
  Valid result:
  recall@10 : 0.1959    mrr@10 : 0.1492    ndcg@10 : 0.1352    hit@10 : 0.3067    precision@10 : 0.0387
  Test result:
  recall@10 : 0.2015    mrr@10 : 0.1624    ndcg@10 : 0.1447    hit@10 : 0.3131    precision@10 : 0.0409

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2001    mrr@10 : 0.1527    ndcg@10 : 0.1385    hit@10 : 0.3121    precision@10 : 0.0396
  Test result:
  recall@10 : 0.2064    mrr@10 : 0.1661    ndcg@10 : 0.1481    hit@10 : 0.3195    precision@10 : 0.0418

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2011    mrr@10 : 0.1523    ndcg@10 : 0.1384    hit@10 : 0.3132    precision@10 : 0.0396
  Test result:
  recall@10 : 0.2051    mrr@10 : 0.1629    ndcg@10 : 0.1462    hit@10 : 0.3176    precision@10 : 0.0415

  learning_rate:0.007
  Valid result:
  recall@10 : 0.1992    mrr@10 : 0.151    ndcg@10 : 0.1372    hit@10 : 0.3107    precision@10 : 0.0393
  Test result:
  recall@10 : 0.2049    mrr@10 : 0.166    ndcg@10 : 0.1474    hit@10 : 0.3182    precision@10 : 0.0414

  learning_rate:0.0001
  Valid result:
  recall@10 : 0.1997    mrr@10 : 0.1511    ndcg@10 : 0.1368    hit@10 : 0.3123    precision@10 : 0.0396
  Test result:
  recall@10 : 0.2053    mrr@10 : 0.1662    ndcg@10 : 0.1476    hit@10 : 0.3186    precision@10 : 0.0418

  learning_rate:0.005
  Valid result:
  recall@10 : 0.2028    mrr@10 : 0.1549    ndcg@10 : 0.1402    hit@10 : 0.3164    precision@10 : 0.0398
  Test result:
  recall@10 : 0.208    mrr@10 : 0.1686    ndcg@10 : 0.1496    hit@10 : 0.3234    precision@10 : 0.0419

  learning_rate:5e-05
  Valid result:
  recall@10 : 0.2002    mrr@10 : 0.1517    ndcg@10 : 0.1375    hit@10 : 0.3141    precision@10 : 0.0398
  Test result:
  recall@10 : 0.2057    mrr@10 : 0.1665    ndcg@10 : 0.1481    hit@10 : 0.3185    precision@10 : 0.0418
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [2:01:10<00:00, 1038.60s/trial, best loss: -0.1402]
  best params:  {'learning_rate': 0.005}
  best result: 
  {'model': 'RecVAE', 'best_valid_score': 0.1402, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2028), ('mrr@10', 0.1549), ('ndcg@10', 0.1402), ('hit@10', 0.3164), ('precision@10', 0.0398)]), 'test_result': OrderedDict([('recall@10', 0.208), ('mrr@10', 0.1686), ('ndcg@10', 0.1496), ('hit@10', 0.3234), ('precision@10', 0.0419)])}
  ```
