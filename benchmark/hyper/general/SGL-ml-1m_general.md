# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [SGL](https://recbole.io/docs/user_guide/model/general/sgl.html)

- **Time cost**: 2043.08s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml                                                            
  ssl_tau choice [0.1,0.2,0.5] 
  drop_ratio choice [0.1,0.2,0.4,0.5] 
  ssl_weight choice [0.05,0.1,0.5]
  ```

- **Best parameters**:

  ```yaml
  ssl_tau: 0.5  
  drop_ratio: 0.1  
  ssl_weight: 0.05
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  drop_ratio:0.1, ssl_tau:0.5, ssl_weight:0.1
  Valid result:
  recall@10 : 0.1651    mrr@10 : 0.3657    ndcg@10 : 0.2037    hit@10 : 0.7139    precision@10 : 0.1515
  Test result:
  recall@10 : 0.1815    mrr@10 : 0.4307    ndcg@10 : 0.2459    hit@10 : 0.7341    precision@10 : 0.1822

  drop_ratio:0.4, ssl_tau:0.5, ssl_weight:0.05
  Valid result:
  recall@10 : 0.1656    mrr@10 : 0.3644    ndcg@10 : 0.2038    hit@10 : 0.7127    precision@10 : 0.1511
  Test result:
  recall@10 : 0.1813    mrr@10 : 0.4332    ndcg@10 : 0.2466    hit@10 : 0.7318    precision@10 : 0.1818

  drop_ratio:0.1, ssl_tau:0.2, ssl_weight:0.05
  Valid result:
  recall@10 : 0.1648    mrr@10 : 0.3693    ndcg@10 : 0.2045    hit@10 : 0.7106    precision@10 : 0.1508
  Test result:
  recall@10 : 0.1808    mrr@10 : 0.4293    ndcg@10 : 0.2454    hit@10 : 0.7303    precision@10 : 0.1814
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [20:25:50<00:00, 2043.08s/trial, best loss: -0.206]
  best params:  {'drop_ratio': 0.1, 'ssl_tau': 0.5, 'ssl_weight': 0.05}
  best result: 
  {'model': 'SGL', 'best_valid_score': 0.206, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1691), ('mrr@10', 0.3663), ('ndcg@10', 0.206), ('hit@10', 0.7192), ('precision@10', 0.1523)]), 'test_result': OrderedDict([('recall@10', 0.1854), ('mrr@10', 0.4341), ('ndcg@10', 0.2492), ('hit@10', 0.7371), ('precision@10', 0.1835)])}
  ```
