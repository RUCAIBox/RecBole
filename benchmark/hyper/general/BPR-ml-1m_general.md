# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [BPR](https://recbole.io/docs/user_guide/model/general/bpr.html)

- **Time cost**: 916.86s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,7e-4,1e-3,5e-3,7e-3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.1558    mrr@10 : 0.3591    ndcg@10 : 0.1982    hit@10 : 0.6922    precision@10 : 0.1483
  Test result:
  recall@10 : 0.1725    mrr@10 : 0.4221    ndcg@10 : 0.2407    hit@10 : 0.7131    precision@10 : 0.1802

  learning_rate:0.007
  Valid result:
  recall@10 : 0.1317    mrr@10 : 0.325     ndcg@10 : 0.1718    hit@10 : 0.6498    precision@10 : 0.1305
  Test result:
  recall@10 : 0.1463    mrr@10 : 0.3689    ndcg@10 : 0.2025    hit@10 : 0.6637    precision@10 : 0.1569

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1576    mrr@10 : 0.3565    ndcg@10 : 0.1982    hit@10 : 0.6965    precision@10 : 0.1489
  Test result:
  recall@10 : 0.1734    mrr@10 : 0.4174    ndcg@10 : 0.2396    hit@10 : 0.7149    precision@10 : 0.1803
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [1:17:57<00:00, 668.15s/trial, best loss: -0.1982]
  best params:  {'learning_rate': 0.0001}
  best result: 
  {'model': 'BPR', 'best_valid_score': 0.1982, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1558), ('mrr@10', 0.3591), ('ndcg@10', 0.1982), ('hit@10', 0.6922), ('precision@10', 0.1483)]), 'test_result': OrderedDict([('recall@10', 0.1725), ('mrr@10', 0.4221), ('ndcg@10', 0.2407), ('hit@10', 0.7131), ('precision@10', 0.1802)])}
  ```
