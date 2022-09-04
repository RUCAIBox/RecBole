# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [SRGNN](https://recbole.io/docs/user_guide/model/sequential/srgnn.html)

- **Time cost**: 29476.25s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001
  Valid result:
  recall@10 : 0.2505    mrr@10 : 0.1101    ndcg@10 : 0.1429    hit@10 : 0.2505    precision@10 : 0.025
  Test result:
  recall@10 : 0.239    mrr@10 : 0.1043    ndcg@10 : 0.1358    hit@10 : 0.239    precision@10 : 0.0239

  learning_rate:0.005
  Valid result:
  recall@10 : 0.2533    mrr@10 : 0.1116    ndcg@10 : 0.1447    hit@10 : 0.2533    precision@10 : 0.0253
  Test result:
  recall@10 : 0.2425    mrr@10 : 0.1035    ndcg@10 : 0.136    hit@10 : 0.2425    precision@10 : 0.0242

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2491    mrr@10 : 0.1099    ndcg@10 : 0.1424    hit@10 : 0.2491    precision@10 : 0.0249
  Test result:
  recall@10 : 0.2382    mrr@10 : 0.1056    ndcg@10 : 0.1365    hit@10 : 0.2382    precision@10 : 0.0238

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2473    mrr@10 : 0.1073    ndcg@10 : 0.1398    hit@10 : 0.2473    precision@10 : 0.0247
  Test result:
  recall@10 : 0.2389    mrr@10 : 0.1029    ndcg@10 : 0.1346    hit@10 : 0.2389    precision@10 : 0.0239

  learning_rate:0.003
  Valid result:
  recall@10 : 0.2553    mrr@10 : 0.1086    ndcg@10 : 0.1428    hit@10 : 0.2553    precision@10 : 0.0255
  Test result:
  recall@10 : 0.2448    mrr@10 : 0.1062    ndcg@10 : 0.1385    hit@10 : 0.2448    precision@10 : 0.0245
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [40:56:21<00:00, 29476.25s/trial, best loss: -0.1447]
  best params:  {'learning_rate': 0.005}
  best result: 
  {'model': 'SRGNN', 'best_valid_score': 0.1447, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2533), ('mrr@10', 0.1116), ('ndcg@10', 0.1447), ('hit@10', 0.2533), ('precision@10', 0.0253)]), 'test_result': OrderedDict([('recall@10', 0.2425), ('mrr@10', 0.1035), ('ndcg@10', 0.136), ('hit@10', 0.2425), ('precision@10', 0.0242)])}
  ```
