# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [FPMC](https://recbole.io/docs/user_guide/model/sequential/fpmc.html)

- **Time cost**: 1471.96s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1842    mrr@10 : 0.0686    ndcg@10 : 0.0955    hit@10 : 0.1842    precision@10 : 0.0184
  Test result:
  recall@10 : 0.1667    mrr@10 : 0.0636    ndcg@10 : 0.0876    hit@10 : 0.1667    precision@10 : 0.0167

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1863    mrr@10 : 0.068    ndcg@10 : 0.0955    hit@10 : 0.1863    precision@10 : 0.0186
  Test result:
  recall@10 : 0.1667    mrr@10 : 0.064    ndcg@10 : 0.0879    hit@10 : 0.1667    precision@10 : 0.0167

  learning_rate:0.003
  Valid result:
  recall@10 : 0.1757    mrr@10 : 0.0635    ndcg@10 : 0.0895    hit@10 : 0.1757    precision@10 : 0.0176
  Test result:
  recall@10 : 0.1573    mrr@10 : 0.0609    ndcg@10 : 0.0833    hit@10 : 0.1573    precision@10 : 0.0157

  learning_rate:0.005
  Valid result:
  recall@10 : 0.1628    mrr@10 : 0.0608    ndcg@10 : 0.0846    hit@10 : 0.1628    precision@10 : 0.0163
  Test result:
  recall@10 : 0.1518    mrr@10 : 0.0558    ndcg@10 : 0.0781    hit@10 : 0.1518    precision@10 : 0.0152

  learning_rate:0.001
  Valid result:
  recall@10 : 0.184    mrr@10 : 0.0671    ndcg@10 : 0.0943    hit@10 : 0.184    precision@10 : 0.0184
  Test result:
  recall@10 : 0.1669    mrr@10 : 0.0624    ndcg@10 : 0.0868    hit@10 : 0.1669    precision@10 : 0.0167
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [2:02:39<00:00, 1471.96s/trial, best loss: -0.0955]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'FPMC', 'best_valid_score': 0.0955, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1842), ('mrr@10', 0.0686), ('ndcg@10', 0.0955), ('hit@10', 0.1842), ('precision@10', 0.0184)]), 'test_result': OrderedDict([('recall@10', 0.1667), ('mrr@10', 0.0636), ('ndcg@10', 0.0876), ('hit@10', 0.1667), ('precision@10', 0.0167)])}
  ```
