# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [GCSAN](https://recbole.io/docs/user_guide/model/sequential/gcsan.html)

- **Time cost**: 43143.38s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.2808    mrr@10 : 0.1226    ndcg@10 : 0.1595    hit@10 : 0.2808    precision@10 : 0.0281
  Test result:
  recall@10 : 0.2634    mrr@10 : 0.1114    ndcg@10 : 0.1468    hit@10 : 0.2634    precision@10 : 0.0263

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2846    mrr@10 : 0.1228    ndcg@10 : 0.1604    hit@10 : 0.2846    precision@10 : 0.0285
  Test result:
  recall@10 : 0.265    mrr@10 : 0.1141    ndcg@10 : 0.1494    hit@10 : 0.265    precision@10 : 0.0265

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2873    mrr@10 : 0.1269    ndcg@10 : 0.1642    hit@10 : 0.2873    precision@10 : 0.0287
  Test result:
  recall@10 : 0.2674    mrr@10 : 0.1142    ndcg@10 : 0.15    hit@10 : 0.2674    precision@10 : 0.0267

  learning_rate:0.003
  Valid result:
  recall@10 : 0.2889    mrr@10 : 0.1259    ndcg@10 : 0.164    hit@10 : 0.2889    precision@10 : 0.0289
  Test result:
  recall@10 : 0.2664    mrr@10 : 0.119    ndcg@10 : 0.1535    hit@10 : 0.2664    precision@10 : 0.0266

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2869    mrr@10 : 0.1252    ndcg@10 : 0.1628    hit@10 : 0.2869    precision@10 : 0.0287
  Test result:
  recall@10 : 0.265    mrr@10 : 0.1143    ndcg@10 : 0.1496    hit@10 : 0.265    precision@10 : 0.0265
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [59:55:16<00:00, 43143.38s/trial, best loss: -0.1642]
  best params:  {'learning_rate': 0.001}
  best result: 
  {'model': 'GCSAN', 'best_valid_score': 0.1642, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2873), ('mrr@10', 0.1269), ('ndcg@10', 0.1642), ('hit@10', 0.2873), ('precision@10', 0.0287)]), 'test_result': OrderedDict([('recall@10', 0.2674), ('mrr@10', 0.1142), ('ndcg@10', 0.15), ('hit@10', 0.2674), ('precision@10', 0.0267)])}
  ```
