# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [EASE](https://recbole.io/docs/user_guide/model/general/ease.html)

- **Time cost**: 23.85s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  reg_weight choice [1.0,10.0,100.0,500.0,1000.0,2000.0]
  ```

- **Best parameters**:

  ```yaml
   reg_weight: 1000.0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  reg_weight:1.0
  Valid result:
  recall@10 : 0.1254    mrr@10 : 0.2919    ndcg@10 : 0.1492    hit@10 : 0.6075    precision@10 : 0.1052
  Test result:
  recall@10 : 0.136     mrr@10 : 0.3325    ndcg@10 : 0.1699    hit@10 : 0.642     precision@10 : 0.1193

  reg_weight:500.0
  Valid result:
  recall@10 : 0.1891    mrr@10 : 0.3985    ndcg@10 : 0.2294    hit@10 : 0.7495    precision@10 : 0.1666
  Test result:
  recall@10 : 0.2149    mrr@10 : 0.4813    ndcg@10 : 0.2864    hit@10 : 0.7852    precision@10 : 0.2074

  reg_weight:1000.0
  Valid result:
  recall@10 : 0.1874    mrr@10 : 0.4017    ndcg@10 : 0.2308    hit@10 : 0.7481    precision@10 : 0.1681
  Test result:
  recall@10 : 0.2139    mrr@10 : 0.4808    ndcg@10 : 0.2865    hit@10 : 0.7837    precision@10 : 0.2082
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 6/6 [02:23<00:00, 23.85s/trial, best loss: -0.2308]
  best params:  {'reg_weight': 1000.0}
  best result: 
  {'model': 'EASE', 'best_valid_score': 0.2308, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1874), ('mrr@10', 0.4017), ('ndcg@10', 0.2308), ('hit@10', 0.7481), ('precision@10', 0.1681)]), 'test_result': OrderedDict([('recall@10', 0.2139), ('mrr@10', 0.4808), ('ndcg@10', 0.2865), ('hit@10', 0.7837), ('precision@10', 0.2082)])}
  ```
