# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [BPR](https://recbole.io/docs/user_guide/model/general/bpr.html)

- **Time cost**: 159.82s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,7e-4,1e-3,5e-3,7e-3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.007
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.1762    mrr@10 : 0.3652    ndcg@10 : 0.2071    hit@10 : 0.7303    precision@10 : 0.1495
  Test result:
  recall@10 : 0.1967    mrr@10 : 0.4236    ndcg@10 : 0.2481    hit@10 : 0.756     precision@10 : 0.1803

  learning_rate:0.007
  Valid result:
  recall@10 : 0.1787    mrr@10 : 0.3643    ndcg@10 : 0.2087    hit@10 : 0.7361    precision@10 : 0.1521
  Test result:
  recall@10 : 0.1995    mrr@10 : 0.4292    ndcg@10 : 0.2523    hit@10 : 0.765     precision@10 : 0.1839

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1713    mrr@10 : 0.3539    ndcg@10 : 0.2024    hit@10 : 0.7181    precision@10 : 0.1491
  Test result:
  recall@10 : 0.188     mrr@10 : 0.4107    ndcg@10 : 0.2397    hit@10 : 0.7364    precision@10 : 0.1768
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [18:38<00:00, 159.82s/trial, best loss: -0.2087]
  best params:  {'learning_rate': 0.007}
  best result: 
  {'model': 'MultiVAE', 'best_valid_score': 0.2087, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1787), ('mrr@10', 0.3643), ('ndcg@10', 0.2087), ('hit@10', 0.7361), ('precision@10', 0.1521)]), 'test_result': OrderedDict([('recall@10', 0.1995), ('mrr@10', 0.4292), ('ndcg@10', 0.2523), ('hit@10', 0.765), ('precision@10', 0.1839)])}
  ```
