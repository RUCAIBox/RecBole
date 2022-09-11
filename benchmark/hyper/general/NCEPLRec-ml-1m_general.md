# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [NCEPLRec](https://recbole.io/docs/user_guide/model/general/nceplrec.html)

- **Time cost**: 2584.78s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  rank choice [100,200,450] 
  beta choice [0.8,1.0,1.3] 
  reg_weight choice [1e-4,1e-2,1e2,15000]
  ```

- **Best parameters**:

  ```yaml
  rank: 100  
  beta: 1.0  
  reg_weight: 15000
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  beta:1.0, rank:200, reg_weight:0.0001
  Valid result:
  recall@10 : 0.1822    mrr@10 : 0.3797    ndcg@10 : 0.2137    hit@10 : 0.7393    precision@10 : 0.1528
  Test result:
  recall@10 : 0.2048    mrr@10 : 0.4529    ndcg@10 : 0.2566    hit@10 : 0.7676    precision@10 : 0.1803

  beta:1.0, rank:100, reg_weight:15000
  Valid result:
  recall@10 : 0.1848    mrr@10 : 0.3939    ndcg@10 : 0.2243    hit@10 : 0.7423    precision@10 : 0.1635
  Test result:
  recall@10 : 0.2077    mrr@10 : 0.4732    ndcg@10 : 0.2748    hit@10 : 0.7709    precision@10 : 0.1985

  beta:1.3, rank:100, reg_weight:100.0
  Valid result:
  recall@10 : 0.1535    mrr@10 : 0.3557    ndcg@10 : 0.1959    hit@10 : 0.688     precision@10 : 0.1482
  Test result:
  recall@10 : 0.1682    mrr@10 : 0.4222    ndcg@10 : 0.2379    hit@10 : 0.7069    precision@10 : 0.1777
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [25:50:52<00:00, 2584.78s/trial, best loss: -0.2243]
  best params:  {'beta': 1.0, 'rank': 100, 'reg_weight': 15000}
  best result: 
  {'model': 'NCEPLRec', 'best_valid_score': 0.2243, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1848), ('mrr@10', 0.3939), ('ndcg@10', 0.2243), ('hit@10', 0.7423), ('precision@10', 0.1635)]), 'test_result': OrderedDict([('recall@10', 0.2077), ('mrr@10', 0.4732), ('ndcg@10', 0.2748), ('hit@10', 0.7709), ('precision@10', 0.1985)])}
  ```
