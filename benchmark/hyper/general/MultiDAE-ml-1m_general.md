# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [MultiDAE](https://recbole.io/docs/user_guide/model/general/multidae.html)

- **Time cost**: 100.87s/trial

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
  learning_rate:0.001
  Valid result:
  recall@10 : 0.1662    mrr@10 : 0.3617    ndcg@10 : 0.2019    hit@10 : 0.7074    precision@10 : 0.1481
  Test result:
  recall@10 : 0.1828    mrr@10 : 0.4142    ndcg@10 : 0.2382    hit@10 : 0.7265    precision@10 : 0.1757

  learning_rate:0.007
  Valid result:
  recall@10 : 0.1785    mrr@10 : 0.362     ndcg@10 : 0.2054    hit@10 : 0.7368    precision@10 : 0.1479
  Test result:
  recall@10 : 0.1972    mrr@10 : 0.4125    ndcg@10 : 0.2425    hit@10 : 0.7568    precision@10 : 0.1769

  learning_rate:0.0007
  Valid result:
  recall@10 : 0.1717    mrr@10 : 0.3622    ndcg@10 : 0.2042    hit@10 : 0.7187    precision@10 : 0.1496
  Test result:
  recall@10 : 0.1875    mrr@10 : 0.4151    ndcg@10 : 0.2408    hit@10 : 0.7384    precision@10 : 0.177
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [11:46<00:00, 100.87s/trial, best loss: -0.2054]
  best params:  {'learning_rate': 0.007}
  best result: 
  {'model': 'MultiDAE', 'best_valid_score': 0.2054, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1785), ('mrr@10', 0.362), ('ndcg@10', 0.2054), ('hit@10', 0.7368), ('precision@10', 0.1479)]), 'test_result': OrderedDict([('recall@10', 0.1972), ('mrr@10', 0.4125), ('ndcg@10', 0.2425), ('hit@10', 0.7568), ('precision@10', 0.1769)])}
  ```
